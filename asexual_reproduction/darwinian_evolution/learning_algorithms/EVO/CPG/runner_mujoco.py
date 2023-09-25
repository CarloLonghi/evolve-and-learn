import concurrent.futures
import math
import os
import tempfile
from typing import List, Optional

import cv2
import mujoco
import mujoco_viewer
import numpy as np
import numpy.typing as npt
from learning_algorithms.EVO.CPG.vision import OpenGLVision
import glfw

try:
    import logging

    old_len = len(logging.root.handlers)

    from dm_control import mjcf

    new_len = len(logging.root.handlers)

    assert (
        old_len + 1 == new_len
    ), "dm_control not adding logging handler as expected. Maybe they fixed their annoying behaviour? https://github.com/deepmind/dm_control/issues/314https://github.com/deepmind/dm_control/issues/314"

    logging.root.removeHandler(logging.root.handlers[-1])
except Exception as e:
    print("Failed to fix absl logging bug", e)
    pass

from pyrr import Quaternion, Vector3
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    BatchResults,
    Environment,
    EnvironmentResults,
    EnvironmentState,
    RecordSettings,
    Runner,
    geometry,
)


class LocalRunner(Runner):
    """Runner for simulating using Mujoco."""

    _headless: bool
    _start_paused: bool
    _num_simulators: int
    _target_points: List

    def __init__(
        self,
        headless: bool = False,
        start_paused: bool = False,
        num_simulators: int = 1,
        target_points: List = [0., 0.],
    ):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        :param start_paused: If True, start the simulation paused. Only possible when not in headless mode.
        :param num_simulators: The number of simulators to deploy in parallel. They will take one core each but will share space on the main python thread for calculating control.
        """
        assert (
            headless or num_simulators == 1
        ), "Cannot have parallel simulators when visualizing."

        assert not (
            headless and start_paused
        ), "Cannot start simulation paused in headless mode."

        self._headless = headless
        self._start_paused = start_paused
        self._num_simulators = num_simulators
        self._target_points = target_points

    @classmethod
    def _run_environment(
        cls,
        env_index: int,
        env_descr: Environment,
        headless: bool,
        record_settings: Optional[RecordSettings],
        start_paused: bool,
        control_step: float,
        sample_step: float,
        simulation_time: int,
        target_points: List
    ) -> EnvironmentResults:
        logging.info(f"Environment {env_index}")

        targets_points = target_points
        target_counter = 0

        model = cls._make_model(env_descr, target_points)

        # TODO initial dof state
        data = mujoco.MjData(model)

        vision_obj = OpenGLVision(model, (35, 20)) # aspect_ratio = tan(h_fov * 0.5)/tan(v_fov * 0.5)
        env_descr.controller.set_picture_w(35)

        initial_targets = [
            dof_state
            for posed_actor in env_descr.actors
            for dof_state in posed_actor.dof_states
        ]
        cls._set_dof_targets(data, initial_targets)

        for posed_actor in env_descr.actors:
            posed_actor.dof_states

        if not headless or record_settings is not None:
            # delegate rendering to the egl back-end
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)

            viewer = mujoco_viewer.MujocoViewer(
                model,
                data,
            )
            viewer._render_every_frame = False  # Private but functionality is not exposed and for now it breaks nothing.
            viewer._paused = start_paused

        if record_settings is not None:
            video_step = 1 / record_settings.fps
            video_file_path = f"{record_settings.video_directory}/{env_index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                video_file_path,
                fourcc,
                record_settings.fps,
                (viewer.viewport.width, viewer.viewport.height),
            )

            viewer._hide_menu = True

        last_control_time = 0.0
        last_sample_time = 0.0
        last_video_time = 0.0  # time at which last video frame was saved

        results = EnvironmentResults([])

        # sample initial state
        results.environment_states.append(
            EnvironmentState(0.0, cls._get_actor_states(env_descr, data, model))
        )
        current_pos = results.environment_states[-1].actor_states[0].position
        save_pos = True
        while (time := data.time) < simulation_time:
            # do control if it is time
            if time >= last_control_time + control_step:
                last_control_time = math.floor(time / control_step) * control_step
                control_user = ActorControl()
                current_pos = results.environment_states[-1].actor_states[0].position
                current_vision = vision_obj.process(model, data)
                env_descr.controller.control(control_step, control_user, current_vision, data.xanchor, current_pos, save_pos)
                actor_targets = control_user._dof_targets
                actor_targets.sort(key=lambda t: t[0])
                targets = [
                    target
                    for actor_target in actor_targets
                    for target in actor_target[1]
                ]
                cls._set_dof_targets(data, targets)
                save_pos = False

            # sample state if it is time
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                results.environment_states.append(
                    EnvironmentState(
                        time, cls._get_actor_states(env_descr, data, model)
                    )
                )

            # step simulation
            mujoco.mj_step(model, data)

            # render if not headless. also render when recording and if it time for a new video frame.
            if not headless or (
                record_settings is not None and time >= last_video_time + video_step
            ):
                viewer.render()

            # capture video frame if it's time
            if record_settings is not None and time >= last_video_time + video_step:
                last_video_time = int(time / video_step) * video_step

                # https://github.com/deepmind/mujoco/issues/285 (see also record.cc)
                img: npt.NDArray[np.uint8] = np.empty(
                    (viewer.viewport.height, viewer.viewport.width, 3),
                    dtype=np.uint8,
                )

                mujoco.mjr_readPixels(
                    rgb=img,
                    depth=None,
                    viewport=viewer.viewport,
                    con=viewer.ctx,
                )
                img = np.flip(img, axis=0)  # img is upside down initially
                img = np.stack((img[:,:,2],img[:,:,1],img[:,:,0]), axis=-1) # switch color channels
                video.write(img)

            reached_target = cls.update_targets_color(model, current_pos, targets_points, target_counter)
            if reached_target:
                target_counter += 1

        if not headless or record_settings is not None:
            viewer.close()

        if record_settings is not None:
            video.release()

        # sample one final time
        results.environment_states.append(
            EnvironmentState(time, cls._get_actor_states(env_descr, data, model))
        )

        return results

    @staticmethod
    def update_targets_color(model: mujoco.MjModel, robot_pos, targets: List[float], target_counter: int) -> bool:
        transparent = np.array([0.9, 0.9, 0.9, .0])
        target_color = np.array([0.9, 0., 0., 1.0])

        reached_target = False
        
        if target_counter < len(targets):
            next_target = targets[target_counter]
            dist = math.sqrt((robot_pos[0] - next_target[0])**2 + (robot_pos[1] - next_target[1])**2)
            if dist <= 0.1:
                model.geom_rgba[1  + target_counter] = transparent
                if target_counter < len(targets) - 1:
                    model.geom_rgba[1  + target_counter + 1] = target_color
                reached_target = True
        return reached_target

    async def run_batch(
        self, batch: Batch, record_settings: Optional[RecordSettings] = None
    ) -> BatchResults:
        """
        Run the provided batch by simulating each contained environment.

        :param batch: The batch to run.
        :param record_settings: Optional settings for recording the runnings. If None, no recording is made.
        :returns: List of simulation states in ascending order of time.
        """
        logging.info("Starting simulation batch with mujoco.")

        control_step = 1 / batch.control_frequency
        sample_step = 1 / batch.sampling_frequency

        if record_settings is not None:
            os.makedirs(record_settings.video_directory, exist_ok=False)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._num_simulators
        ) as executor:
            futures = [
                executor.submit(
                    self._run_environment,
                    env_index,
                    env_descr,
                    self._headless,
                    record_settings,
                    self._start_paused,
                    control_step,
                    sample_step,
                    batch.simulation_time,
                    self._target_points
                )
                for env_index, env_descr in enumerate(batch.environments)
            ]
            results = BatchResults([future.result() for future in futures])

        logging.info("Finished batch.")

        return results

    @staticmethod
    def _make_model(env_descr: Environment, target_points: List) -> mujoco.MjModel:
        env_mjcf = mjcf.RootElement(model="environment")

        env_mjcf.compiler.angle = "radian"

        env_mjcf.option.timestep = 0.0005
        env_mjcf.option.integrator = "RK4"

        env_mjcf.option.gravity = [0, 0, -9.81]

        env_mjcf.size.nconmax = 150

        env_mjcf.worldbody.add(
            "light",
            pos=[0, 0, 100],
            ambient=[0.5, 0.5, 0.5],
            directional=True,
            castshadow=False,
        )
        env_mjcf.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1=".2 .2 .2",
            rgb2=".3 .3 .3",
        )
        env_mjcf.asset.add(
            "material",
            name="grid",
            texture="grid",
            texrepeat="1 1",
            texuniform="true",
            reflectance=".2"
        )
        heightmaps: List[geometry.Heightmap] = []
        for geo in env_descr.static_geometries:
            if isinstance(geo, geometry.Plane):
                env_mjcf.worldbody.add(
                    "geom",
                    type="plane",
                    pos=[geo.position.x, geo.position.y, geo.position.z],
                    size=[geo.size.x / 2.0, geo.size.y / 2.0, 1.0],
                    rgba=[1., 1., 1., 1.0],
                    material="grid",
                )
            elif isinstance(geo, geometry.Heightmap):
                env_mjcf.asset.add(
                    "hfield",
                    name=f"hfield_{len(heightmaps)}",
                    nrow=len(geo.heights),
                    ncol=len(geo.heights[0]),
                    size=[geo.size.x, geo.size.y, geo.size.z, geo.base_thickness],
                )

                env_mjcf.worldbody.add(
                    "geom",
                    type="hfield",
                    hfield=f"hfield_{len(heightmaps)}",
                    pos=[geo.position.x, geo.position.y, geo.position.z],
                    quat=[
                        geo.orientation.x,
                        geo.orientation.y,
                        geo.orientation.z,
                        geo.orientation.w,
                    ],
                    # size=[geo.size.x, geo.size.y, 1.0],
                    rgba=[geo.color.x, geo.color.y, geo.color.z, 1.0],
                )
                heightmaps.append(geo)
            else:
                raise NotImplementedError()

        # add target points markers
        target_points = target_points
        i = 0
        env_mjcf.worldbody.add(
            "geom",
            name="target_point_"+str(i),
            pos=[target_points[0][0], target_points[0][1], 0.005],
            size=[0.1, 0.000001],
            type="cylinder",
            condim=1,
            contype=2,
            conaffinity=2,
            rgba=".9 0. 0. 1.",
        )
        i += 1
        for point in target_points[1:]:
           env_mjcf.worldbody.add(
               "geom",
               name="target_point_"+str(i),
               pos=[point[0], point[1], 0.005],
               size=[0.1, 0.000001],
               type="cylinder",
               condim=1,
               contype=2,
               conaffinity=2,
               rgba=".9 .9 .9 0.",
           )
           i+= 1
        env_mjcf.visual.headlight.active = 0

        for actor_index, posed_actor in enumerate(env_descr.actors):
            urdf = physbot_to_urdf(
                posed_actor.actor,
                f"robot_{actor_index}",
                Vector3(),
                Quaternion(),
            )

            model = mujoco.MjModel.from_xml_string(urdf)

            # mujoco can only save to a file, not directly to string,
            # so we create a temporary file.
            try:
                with tempfile.NamedTemporaryFile(
                    mode="r+", delete=True, suffix="_mujoco.urdf"
                ) as botfile:
                    mujoco.mj_saveLastXML(botfile.name, model)
                    robot = mjcf.from_file(botfile)
            # handle an exception when the xml saving fails, it's almost certain to occur on Windows
            # since NamedTemporaryFile can't be opened twice when the file is still open.
            except Exception as e:
                print(repr(e))
                print(
                    "Setting 'delete' parameter to False so that the xml can be saved"
                )
                with tempfile.NamedTemporaryFile(
                    mode="r+", delete=False, suffix="_mujoco.urdf"
                ) as botfile:
                    # to make sure the temp file is always deleted,
                    # an error catching is needed, in case the xml saving fails and crashes the program
                    try:
                        mujoco.mj_saveLastXML(botfile.name, model)
                        robot = mjcf.from_file(botfile)
                        # On Windows, an open file can’t be deleted, and hence it has to be closed first before removing
                        botfile.close()
                        os.remove(botfile.name)
                    except Exception as e:
                        print(repr(e))
                        # On Windows, an open file can’t be deleted, and hence it has to be closed first before removing
                        botfile.close()
                        os.remove(botfile.name)

            LocalRunner._set_parameters(robot)

            fps_cam_pos = [
                0.0,
                0.0,
                0.07
            ]
            robot.worldbody.add("camera", name="vision", mode="fixed", dclass=robot.full_identifier,
                                pos=fps_cam_pos, xyaxes="1 0 0 0 0 1", fovy=37.9)
            robot.worldbody.add('site',
                                name=robot.full_identifier[:-1] + "_camera",
                                pos=fps_cam_pos, rgba=[0, 0, 1, 1],
                                type="ellipsoid", size=[0.0001, 0.025, 0.025],
                                xyaxes="0 -1 0 0 0 1")

            for joint in posed_actor.actor.joints:
                robot.actuator.add(
                    "position",
                    kp=1.0,
                    joint=robot.find(
                        namespace="joint",
                        identifier=joint.name,
                    ),
                )
                robot.actuator.add(
                    "velocity",
                    kv=0.05,
                    joint=robot.find(namespace="joint", identifier=joint.name),
                )

            attachment_frame = env_mjcf.attach(robot)
            attachment_frame.add("freejoint")
            attachment_frame.pos = [
                posed_actor.position.x,
                posed_actor.position.y,
                posed_actor.position.z,
            ]

            attachment_frame.quat = [
                posed_actor.orientation.x,
                posed_actor.orientation.y,
                posed_actor.orientation.z,
                posed_actor.orientation.w,
            ]

        xml = env_mjcf.to_xml_string()
        if not isinstance(xml, str):
            raise RuntimeError("Error generating mjcf xml.")

        model = mujoco.MjModel.from_xml_string(xml)

        # set height map values
        offset = 0

        for heightmap in heightmaps:
            for x in range(len(heightmap.heights)):
                for y in range(len(heightmap.heights[0])):
                    model.hfield_data[
                        y * len(heightmap.heights) + x
                    ] = heightmap.heights[x][y]
            offset += len(heightmap.heights) * len(heightmap.heights[0])
        
        return model
    
    @classmethod
    def _get_actor_states(
        cls, env_descr: Environment, data: mujoco.MjData, model: mujoco.MjModel
    ) -> List[ActorState]:
        return [
            cls._get_actor_state(i, data, model) for i in range(len(env_descr.actors))
        ]

    @staticmethod
    def _get_actor_state(
        robot_index: int, data: mujoco.MjData, model: mujoco.MjModel
    ) -> ActorState:
        bodyid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"robot_{robot_index}/",  # the slash is added by dm_control. ugly but deal with it
        )
        assert bodyid >= 0

        qindex = model.body_jntadr[bodyid]

        # explicitly copy because the Vector3 and Quaternion classes don't copy the underlying structure
        position = Vector3([n for n in data.qpos[qindex : qindex + 3]])
        orientation = Quaternion([n for n in data.qpos[qindex + 3 : qindex + 3 + 4]])

        return ActorState(position, orientation)

    @staticmethod
    def _set_dof_targets(data: mujoco.MjData, targets: List[float]) -> None:
        if len(targets) * 2 != len(data.ctrl):
            raise RuntimeError("Need to set a target for every dof")
        for i, target in enumerate(targets):
            data.ctrl[2 * i] = target
            data.ctrl[2 * i + 1] = 0

    @staticmethod
    def _set_recursive_parameters(element):
        if element.tag == "body":
            for sub_element in element.body._elements:
                LocalRunner._set_recursive_parameters(sub_element)

        if element.tag == "geom":
            element.friction = [0.7, 0.1, 0.1]
            if math.isclose(element.size[0], 0.044, abs_tol=0.001):
                element.rgba = [1., 1., 0., 1.]
            elif math.isclose(element.size[0], 0.031, abs_tol=0.001):
                element.rgba = [.1, 0., 1., 1.]
            else:
                element.rgba = [0., 1., 0., 1.]

    @staticmethod
    def _set_parameters(robot):
        for element in robot.worldbody.body._elements:
            LocalRunner._set_recursive_parameters(element)