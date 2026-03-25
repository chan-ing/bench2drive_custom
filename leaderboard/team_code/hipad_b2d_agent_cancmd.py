import os
import cv2
import json
import time
import math
import copy
import carla
import torch
import pathlib
import datetime
import importlib
import numpy as np

from PIL import Image
from scipy.optimize import fsolve
from pyquaternion import Quaternion
from torchvision import transforms as T

from mmcv import Config
from mmcv.runner import (
    load_checkpoint,
    wrap_fp16_model,
)
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate as mm_collate_to_batch_form

from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

from team_code.planner import RoutePlanner
from team_code.pid_controller import PIDController
from team_code.visualize import draw_bboxes3d
from leaderboard.autoagents import autonomous_agent
from projects.mmdet3d_plugin.datasets.lm_utils.data_utils import preprocess
from projects.mmdet3d_plugin.datasets.lm_utils.constants import DEFAULT_IMAGE_TOKEN


SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

LIDAR2IMG = {
    'CAM_FRONT': np.array([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                           [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                           [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_FRONT_LEFT': np.array([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_FRONT_RIGHT': np.array([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                 [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK': np.array([[-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                          [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                          [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK_LEFT': np.array([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                               [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK_RIGHT': np.array([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
}

LIDAR2CAM = {
    'CAM_FRONT': np.array([[1., 0., 0., 0.],
                           [0., 0., -1., -0.24],
                           [0., 1., 0., -1.19],
                           [0., 0., 0., 1.]]),

    'CAM_FRONT_LEFT': np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                [0., 0., -1., -0.24],
                                [-0.81915204, 0.57357644, 0., -0.82909407],
                                [0., 0., 0., 1.]]),

    'CAM_FRONT_RIGHT': np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                 [0., 0., -1., -0.24],
                                 [0.81915204, 0.57357644, 0., -0.82909407],
                                 [0., 0., 0., 1.]]),

    'CAM_BACK': np.array([[-1., 0., 0., 0.],
                          [0., 0., -1., -0.24],
                          [0., -1., 0., -1.61],
                          [0., 0., 0., 1.]]),

    'CAM_BACK_LEFT': np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                               [0., 0., -1., -0.24],
                               [-0.93969262, -0.34202014, 0., -0.49288953],
                               [0., 0., 0., 1.]]),

    'CAM_BACK_RIGHT': np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                [0., 0., -1., -0.24],
                                [0.93969262, -0.34202014, 0., -0.49288953],
                                [0., 0., 0., 1.]])
}

CAM2IMG = {
    'CAM_FRONT': np.array([[1142.51841 , 0., 800., 0.],
                           [0., 1142.51841, 450., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]]),

    'CAM_FRONT_LEFT': np.array([[1142.51840553, 0., 800., 0.],
                                [0., 1142.51841, 450., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]]),

    'CAM_FRONT_RIGHT': np.array([[1142.51841061, 0., 800, 0.],
                                 [0., 1142.51841, 450, 0.],
                                 [0., 0. ,1., 0.],
                                 [0., 0. ,0., 1.]]),

    'CAM_BACK': np.array([[560.166031, 0., 800., 0.],
                          [0. ,560.166031, 450., 0.],
                          [0. , 0., 1., 0.],
                          [0. , 0., 0., 1.]]),

    'CAM_BACK_LEFT': np.array([[1142.51840683, 0., 800., 0.],
                               [0., 1142.51841, 450., 0.],
                               [0., 0. , 1., 0.],
                               [0., 0. , 0., 1.]]),

    'CAM_BACK_RIGHT': np.array([[1142.51841041, 0., 800, 0.],
                                [0., 1142.51841, 450., 0.],
                                [0., 0. , 1., 0.],
                                [0., 0. , 0., 1.]])
}

LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])

unreal2cam = np.array([[0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0],
                               [0.0, 548.993771650447, 256.0, 0],
                               [0.0, 0.0, 1.0, 0],
                               [0, 0, 0, 1.0]])

topdown_extrinsics = np.array([[0.0, -0.0, -1.0, 50.0],
                               [0.0, 1.0, -0.0, 0.0],
                               [1.0, -0.0, 0.0, -0.0],
                               [0.0, 0.0, 0.0, 1.0]])

CAMERA = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

def get_entry_point():
    return 'SparseAgent'


class SparseAgent(autonomous_agent.AutonomousAgent):
    def sensors(self):
        sensors = [
            # camera rgb
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_RIGHT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -2.0, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'CAM_BACK'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_RIGHT'
            },
            # imu
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU'
            },
            # gps
            {
                'type': 'sensor.other.gnss',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'GPS'
            },
            # speed
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'SPEED'
            },
        ]
        if IS_BENCH2DRIVE:
            sensors += [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 60.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 2048, 'height': 2048, 'fov': 4 * 10.0,
                    'id': 'bev'
                }]
        return sensors

    def setup(self, path_to_conf_file):
        self.device = "cuda"
        self.track = autonomous_agent.Track.SENSORS

        self.step = -1
        self.steer_step = 0
        self.last_steer = 0
        self.last_moving_step = -1
        self.last_moving_status = 0
        self.frame_rate = 20

        self.ckpt_path = path_to_conf_file.split('+')[1]
        self.config_path = path_to_conf_file.split('+')[0]
        self.save_name = path_to_conf_file.split('+')[-1]

        self.pidcontroller = PIDController(
            turn_KP=1.,
            turn_KI=0.75,
            turn_KD=0.0,
            turn_n=10,
            speed_KP=5.0,
            speed_KI=0.5,
            speed_KD=1.0,
            speed_n=10,
            waypoint_time=0.5)

        self.wall_start = time.time()
        self.initialized = False
        self.use_bgr_img = True
        self.data_aug_conf = None

        cfg = Config.fromfile(self.config_path)
        cfg.model.head.onedecoder_head.with_close_loop = True # align training frequency
        if hasattr(cfg, 'data_aug_conf'):
            self.data_aug_conf = cfg.data_aug_conf
        if hasattr(cfg.model.head, 'evaluate_bench2dive'):
            cfg.model.head.evaluate_bench2dive = False
        if hasattr(cfg, "plugin") and hasattr(cfg, "plugin_dir") and cfg.plugin:
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

        self.model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        if self.ckpt_path is not None:
            checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu')

        self.model.cuda()
        self.model.eval()
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph', 'LoadMultiViewImageFromFiles']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        #  breakpoint()
        self.use_action_token = cfg.get('use_action_token', False)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.brake = 0.0
        control.throttle = 0.0

        self.prev_control = control
        self.prev_control_cache = []

        self.is_visualize = True
        self.visualize_interval = 2

        if self.is_visualize:
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'metas').mkdir()
            (self.save_path / 'images').mkdir()
        else:
            self.save_path = pathlib.Path(os.environ['SAVE_PATH'])

        self.lidar2cam = LIDAR2CAM
        self.lidar2img = LIDAR2IMG
        self.lidar2ego = LIDAR2EGO
        self.cam2img = CAM2IMG
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0

            def equations(vars):
                x, y = vars
                eq1 = ((lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA)) -
                       math.cos(x * math.pi / 180) * y)
                eq2 = (math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA
                       * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180)
                       * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360)))
                return [eq1, eq2]

            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        # breakpoint()
        self._route_planner.set_route(self._global_plan, True)
        # breakpoint()
        self.initialized = True
        self.metric_info = {}
        self._cache_traj = np.array([
            [0., 1.],
            [0., 2.],
            [0., 3.],
            [0., 4.],
            [0., 5.],
            [0., 6.],
        ], dtype=np.float32)
        self.valid_traj_flag = True
        self.valid_traj_flag_count = 0

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in CAMERA:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps) # pos array([-1.12064885e-01,  3.67479266e+03])
        # breakpoint()
        waypoint_routes = self._route_planner.run_step(pos)
        # breakpoint()

        if len(waypoint_routes) >= 3:
            target_xy = waypoint_routes[1][0]
            target_xy_next = waypoint_routes[2][0]
            command = waypoint_routes[0][1]
            command_next = waypoint_routes[1][1]
        elif len(waypoint_routes) == 2:
            target_xy = target_xy_next = waypoint_routes[1][0]
            command = command_next = waypoint_routes[0][1]
        else:
            target_xy = target_xy_next = waypoint_routes[0][0]
            command = command_next = waypoint_routes[0][1]

        if (math.isnan(compass) == True):  # It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
            'imgs': imgs,
            'gps': gps,
            'pos': pos,
            'bev': bev, # (512, 512, 3)
            'speed': speed,
            'compass': compass,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'target_xy': target_xy,
            'target_xy_next': target_xy_next,
            'command': command,
            'command_next': command_next
        }

        return result

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH)
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    def build_vlm_inputs(self):
        # breakpoint()
        if self.use_action_token :
            sources = [[
                {"from": "human", "value": "Please provide the planning trajectory for the ego car without reasons. "},
                {"from": "gpt", "value": ""},
            ]]
        else:
            sources = [[
                {"from": "human", "value": "Please provide the planning trajectory for the ego car without reasons. "},
                {"from": "gpt", "value": "Here is the planning trajectory [PT, ("},
            ]]

        vlm_labels = [anno[0]["value"] for anno in sources]
        for lm_anno in sources:
            lm_anno[0]["value"] = DEFAULT_IMAGE_TOKEN + "\n" + lm_anno[0]["value"]

        tokenizer = self.model.head.onedecoder_head.tokenizer
        vqa_converted = preprocess(sources, tokenizer, True, False)
        input_ids = vqa_converted["input_ids"]

        for i in range(len(input_ids)):
            if len(input_ids[i]) > 0 and input_ids[i][-1] == 2:
                input_ids[i] = input_ids[i][:-1]
        # breakpoint
        return input_ids, vlm_labels, sources

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self.init()
        # breakpoint()
        tick_data = self.tick(input_data)
        # breakpoint()
        inputs = {}
        inputs['img'] = []
        inputs['folder'] = ''
        inputs['scene_token'] = ''
        inputs['lidar2img'] = []
        inputs['lidar2cam'] = []
        inputs['frame_idx'] = 0
        inputs['timestamp'] = self.step / self.frame_rate
        for cam in CAMERA:
            inputs['lidar2img'].append(self.lidar2img[cam])
            inputs['lidar2cam'].append(self.lidar2cam[cam])
            img = tick_data['imgs'][cam]
            if self.use_bgr_img:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            inputs['img'].append(img)
        inputs['lidar2img'] = np.stack(inputs['lidar2img'], axis=0)
        inputs['lidar2cam'] = np.stack(inputs['lidar2cam'], axis=0)
        inputs['aug_config'] = self.get_augmentation()

        # data
        ego_pos = [tick_data['pos'][0], -tick_data['pos'][1]]
        target_xy = [tick_data['target_xy'][0], -tick_data['target_xy'][1]]
        target_xy_next = [tick_data['target_xy_next'][0], -tick_data['target_xy_next'][1]]

        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi / 2
        ego_theta_degree = ego_theta / np.pi * 180
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        ego_speed = tick_data['speed']
        acceleration = tick_data['acceleration']
        ego_accel = [acceleration[0], -acceleration[1], acceleration[2]]
        angular_velocity = -tick_data['angular_velocity']

        # status
        custom_status = np.zeros(6)
        custom_status[0] = ego_speed
        custom_status[1:3] = ego_accel[:2]
        custom_status[3:5] = angular_velocity[:2]
        custom_status[5] = self.pid_metadata['steer'] if hasattr(self, 'pid_metadata') else 0
        inputs['custom_status'] = np.array(custom_status, np.float32)

        # breakpoint()
        # command
        command = tick_data['command']
        command_next = tick_data['command_next']
        if command < 0: command = 4
        command -= 1
        command_onehot = np.zeros(6, np.float32)
        command_onehot[command] = 1
        inputs['command'] = command
        inputs['gt_ego_fut_cmd'] = command_onehot # array([0., 0., 0., 1., 0., 0.], dtype=float32)
        
        # breakpoint()
        # target point
        target_xy = np.array([target_xy[0] - ego_pos[0], target_xy[1] - ego_pos[1]])
        target_xy_next = np.array([target_xy_next[0] - ego_pos[0], target_xy_next[1] - ego_pos[1]])
        # breakpoint()

        theta_to_lidar = raw_theta
        rotation_matrix = np.array([[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)],
                                    [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])

        target_point = np.array(rotation_matrix @ target_xy, dtype=np.float32)
        target_point_next = np.array(rotation_matrix @ target_xy_next, dtype=np.float32)

        inputs['target_point'] = target_point
        inputs['target_point_next'] = target_point_next
        
        canbus_cmd = np.zeros(18, dtype=np.float32)
        canbus_cmd[0] = ego_speed
        canbus_cmd[1:4] = ego_accel
        canbus_cmd[4:7] = angular_velocity
        canbus_cmd[7] = self.pid_metadata['steer'] if hasattr(self, 'pid_metadata') else 0.0
        canbus_cmd[8:10] = target_point
        canbus_cmd[10:12] = target_point_next
        canbus_cmd[12:18] = command_onehot
        inputs['canbus'] = canbus_cmd[:12]
        inputs['canbus_cmd'] = canbus_cmd[:18]
        inputs['canbus_cmd_wo_ego'] = canbus_cmd[8:] # 10
        
        # metas
        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = ego_pos
        
        lidar2global = ego2world @ self.lidar2ego
        inputs['l2g_r_mat'] = lidar2global[0:3, 0:3]
        inputs['l2g_t'] = lidar2global[0:3, 3]
        inputs['lidar2global'] = lidar2global
        image_h, image_w = self.data_aug_conf['final_dim']
        inputs['image_wh'] = np.array([[image_w, image_h] for _ in CAMERA], dtype=np.float32)
        input_ids, vlm_labels, planning_texts = self.build_vlm_inputs()
        inputs['input_ids'] = input_ids
        inputs['vlm_labels'] = vlm_labels
        inputs['planning_texts'] = planning_texts
        inputs['save_idx'] = f'{self.save_name}_{self.step:05d}'
        # breakpoint()

        inputs = self.inference_only_pipeline(inputs)
        # Keep VLM/meta fields as cpu_only containers so mmcv collate
        # does not recurse into python strings/lists.
        if not isinstance(inputs.get('input_ids'), DataContainer):
            inputs['input_ids'] = DataContainer(inputs['input_ids'], stack=False, cpu_only=True)
        if not isinstance(inputs.get('vlm_labels'), DataContainer):
            inputs['vlm_labels'] = DataContainer(inputs['vlm_labels'], stack=False, cpu_only=True)
        if not isinstance(inputs.get('planning_texts'), DataContainer):
            inputs['planning_texts'] = DataContainer(inputs['planning_texts'], stack=False, cpu_only=True)
        if not isinstance(inputs.get('save_idx'), DataContainer):
            inputs['save_idx'] = DataContainer(inputs['save_idx'], stack=False, cpu_only=True)
        inputs = mm_collate_to_batch_form([inputs], samples_per_gpu=1)
        for key, value in inputs.items():
            if isinstance(value, DataContainer):
                inputs[key] = value.data[0]
            elif isinstance(value[0], DataContainer):
                inputs[key] = value[0].data
            else:
                inputs[key] = value
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        # breakpoint()
        outputs = self.model(
            img=inputs['img'],
            img_metas=inputs['img_metas'],
            projection_mat=inputs['projection_mat'],
            gt_ego_fut_cmd=inputs['gt_ego_fut_cmd'],
            image_wh=inputs['image_wh'],
            timestamp=inputs['timestamp'],
            target_point=inputs['target_point'],
            input_ids=inputs['input_ids'],
            vlm_labels=inputs['vlm_labels'],
            planning_texts=inputs['planning_texts'],
            save_idx=inputs['save_idx'],
            rescale=True,
            return_loss=False,
        )

        # control
        # plan_temp_name = 'plan_speed_5hz'  # 지금 넣어줄 정보는 2hz vlm_traj
        # plan_spat_name = 'plan_spat_2m'

        pred_temp_traj = None
        # if plan_temp_name in outputs[0]['img_bbox']:
        #     pred_temp_traj = outputs[0]['img_bbox'][plan_temp_name].cpu().numpy()
        #     outputs[0]['img_bbox']['temporal_planning'] = outputs[0]['img_bbox'][plan_temp_name]

        pred_spat_traj = None
        # if plan_spat_name in outputs[0]['img_bbox']:
        #     pred_spat_traj = outputs[0]['img_bbox'][plan_spat_name].cpu().numpy()
        #     outputs[0]['img_bbox']['spatial_planning'] = outputs[0]['img_bbox'][plan_spat_name]
        # breakpoint()
        
        # original VLM_TRAJ
        #vlm_traj = np.array(outputs[0]['img_bbox']['vlm_traj'], dtype=np.float32)
        
        # kclee VLM_LOGITS_TRAJ
        # vlm_traj = np.array(outputs[0]['img_bbox']['vlm_logit_traj'], dtype=np.float32)
        if self.use_action_token :
             vlm_traj = np.array(outputs[0]['img_bbox']['vlm_logit_traj'], dtype=np.float32)
        else:
            vlm_traj = np.array(outputs[0]['img_bbox']['vlm_traj'], dtype=np.float32)

        if (
            not isinstance(vlm_traj, np.ndarray)
            or vlm_traj.ndim != 2
            or vlm_traj.shape != (6, 2)
            or not np.isfinite(vlm_traj).all()
        ):
            self.valid_traj_flag = False
            self.valid_traj_flag_count += 1
            wrong_traj = vlm_traj.copy() if isinstance(vlm_traj, np.ndarray) else vlm_traj
            vlm_traj = self._cache_traj.copy()

            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Warning: Invalid VLM trajectory detected.", self.valid_traj_flag_count, "Times", "\t", "Frame :", self.step)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print()
        else:
            self.valid_traj_flag = True
            self._cache_traj = vlm_traj.copy()
            wrong_traj = None
        # vlm_traj의 길이가 6보다 크면 6으로 자르고, 6보다 작으면 한 번더 try
        # if len(vlm_traj) > 6:
        #     print()
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print("Warning: vlm_traj length is greater than 6, truncating to 6.")
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print()
        #     vlm_traj = vlm_traj[:6]
        # elif len(vlm_traj) < 6:
        #     print()
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print("Warning: vlm_traj length is lower than 6, padding to 6 with zeros.")
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print()
        #     padding = np.zeros((6 - len(vlm_traj), 2), dtype=np.float32)
        #     vlm_traj = np.concatenate([padding, vlm_traj], axis=0)
        pred_temp_traj = vlm_traj
        outputs[0]['img_bbox']['temporal_planning'] = pred_temp_traj
        
        
        # breakpoint()
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(
            pred_temp_traj, pred_spat_traj, ego_speed, target_point)
        # breakpoint()
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        control.steer = np.clip(float(steer_traj), -1, 1) # 이미 PID contoller에서 steer_traj가 -1~1 사이로 나오도록 조정했지만 또 clipping수행.
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)

        # breakpoint()
        self.pid_metadata = metadata_traj
        self.pid_metadata['wrong_traj'] = wrong_traj.tolist() if wrong_traj is not None else None
        self.pid_metadata['agent'] = 'only_traj'
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)         # PID controller에서 나온 steer 값
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan_temp'] = pred_temp_traj.tolist()
        # self.pid_metadata['plan_spat'] = pred_spat_traj.tolist() # spatial traj는 시각화에서만 보여주고 제어에는 사용하지 않음
        self.pid_metadata['command'] = command
        self.pid_metadata['target_point'] = [float(target_point[0]), float(target_point[1])]

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info

        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()
        # breakpoint()
        if self.is_visualize and self.step % self.visualize_interval == 0:
            self.visualize(tick_data, inputs, outputs, pred_temp_traj, target_point, steer_traj, metadata_traj['aim'], metadata_traj['angle_final'], metadata_traj['aim_dist'], metadata_traj['best_norm'], metadata_traj['wrong_traj'])
        # breakpoint()

        self.prev_control = control
        if len(self.prev_control_cache) == 10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control

    def visualize(self, tick_data, input_batch, output_batch, pred_planning, target_point, steer_traj, aim_point, angle_final, aim_dist, best_norm, wrong_traj):
        rw, rh = 960//2, 540//2

        pred_temp_traj = np.asarray(pred_planning, dtype=np.float32)
        pred_planning = np.concatenate([np.array([[0, 0]], dtype=np.float32), pred_temp_traj])
        aim_point = np.asarray(aim_point, dtype=np.float32)
        wrong_traj = np.asarray(wrong_traj, dtype=np.float32) if wrong_traj is not None else None

        def format_panel_lines(name, value):
            if value is None:
                return [f"{name}: None"]
            try:
                arr = np.asarray(value, dtype=np.float32)
                if arr.size == 0:
                    return [f"{name}: []"]
                if arr.ndim == 0:
                    return [f"{name}: {float(arr):.3f}"]
                if arr.ndim == 1:
                    values = ", ".join(f"{float(v):.3f}" for v in arr.tolist())
                    return [f"{name}: [{values}]"]
                lines = [f"{name}:"]
                for row in arr.reshape(arr.shape[0], -1):
                    values = ", ".join(f"{float(v):.3f}" for v in row.tolist())
                    lines.append(f"  [{values}]")
                return lines
            except Exception:
                return [f"{name}: {value}"]

        # theta = angle * (pi / 2)
        # x = r * sin(theta)
        # y = r * cos(theta)
        steer_angle = float(np.clip(steer_traj, -1.0, 1.0)) * (np.pi / 2.0)
        steer_point = np.array([np.sin(steer_angle) * 5.0, np.cos(steer_angle) * 5.0], dtype=np.float32)
        angle_final_angle = float(np.clip(angle_final, -1.0, 1.0)) * (np.pi / 2.0)
        angle_final_point = np.array([np.sin(angle_final_angle) * 5.0, np.cos(angle_final_angle) * 5.0], dtype=np.float32)

        with_spatial_planning = False
        if 'spatial_planning' in output_batch[0]['img_bbox']:
            with_spatial_planning = True
            spatial_planning = output_batch[0]['img_bbox']['spatial_planning'].cpu().numpy()
            spatial_planning = np.concatenate([np.array([[0, 0]]), spatial_planning])

        ### plot img
        # imgs = []
        # for cam in CAMERA:
        #     img = tick_data['imgs'][cam]

        #     # draw agent box on image
        #     if 'boxes_3d' in output_batch[0]['img_bbox']:
        #         pred_bboxes3d, pred_labels3d, pred_trajs3d = self.get_bboxes(output_batch)

        #         img = draw_bboxes3d(img, pred_bboxes3d,
        #                             intrinsic=self.cam2img[cam],
        #                             extrinsic=self.lidar2cam[cam],
        #                             color=(0, 0, 255))

        #     # draw ego traj on image
        #     if cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        #         # spatial traj
        #         if with_spatial_planning:
        #             coord3d = np.concatenate([spatial_planning,
        #                                       np.ones_like(pred_planning[:, :1]) * -1.8,
        #                                       np.ones_like(pred_planning[:, :1])], axis=1)
        #             coord2d = (self.lidar2img[cam] @ coord3d.T).T
        #             coord2d = coord2d[coord2d[:, 2] > 1e-5]
        #             coord2d = coord2d[:, :2] / coord2d[:, 2:3]
        #             img = self.draw_trajectory(img, coord2d, line_color=(0, 200, 255), line_thickness=10,
        #                                        point_color=(0, 200, 255), point_thickness=8, point_radius=8)

        #         # draw steer point # 주황색
                
        #         # steer = -2.9914600331225794e-08
        #         # ego 중심기준 전방 기준, 범위 좌측 90도, 우측 90도 [-1, 1]
        #         # 해당 각도에 해당하는 점을 ego 전방 5m 지점에 찍고 ego 원점점에서 부터 해당 점을 잇는 직선을 시각화
        #         steer_coord3d = np.array([
        #             [0.0, 0.0, -1.8, 1.0],
        #             [steer_point[0], steer_point[1], -1.8, 1.0],
        #         ], dtype=np.float32)
        #         steer_coord2d = (self.lidar2img[cam] @ steer_coord3d.T).T
        #         steer_coord2d = steer_coord2d[:, :2] / np.abs(steer_coord2d[:, 2:3])
        #         img = cv2.line(
        #             img,
        #             (int(steer_coord2d[0, 0]), int(steer_coord2d[0, 1])),
        #             (int(steer_coord2d[1, 0]), int(steer_coord2d[1, 1])),
        #             (255, 165, 0),
        #             10,
        #         )
        #         img = cv2.circle(
        #             img,
        #             (int(steer_coord2d[1, 0]), int(steer_coord2d[1, 1])),
        #             6,
        #             (255, 165, 0),
        #             6,
        #         )

        #         # temporal traj
        #         coord3d = np.concatenate([pred_planning,
        #                                   np.ones_like(pred_planning[:, :1]) * -1.8,
        #                                   np.ones_like(pred_planning[:, :1])], axis=1)
        #         coord2d = (self.lidar2img[cam] @ coord3d.T).T
        #         coord2d = coord2d[coord2d[:, 2] > 1e-5]
        #         coord2d = coord2d[:, :2] / coord2d[:, 2:3]
        #         img = self.draw_trajectory(img, coord2d, line_color=(255, 0, 0), line_thickness=10,
        #                                    point_color=(255, 0, 0), point_thickness=8, point_radius=8)

                    
        #         # angle_final --> draw steer point구현과 동일하게 angle_final에 해당하는 점과 ego 원점을 잇는 직선을 시각화
        #         # angle_final = -0.03489949670250097
                    

        #         angle_final_coord3d = np.array([
        #             [0.0, 0.0, -1.8, 1.0],
        #             [angle_final_point[0], angle_final_point[1], -1.8, 1.0],
        #         ], dtype=np.float32)
        #         angle_final_coord2d = (self.lidar2img[cam] @ angle_final_coord3d.T).T
        #         angle_final_coord2d = angle_final_coord2d[:, :2] / np.abs(angle_final_coord2d[:, 2:3])
        #         img = cv2.line(
        #             img,
        #             (int(angle_final_coord2d[0, 0]), int(angle_final_coord2d[0, 1])),
        #             (int(angle_final_coord2d[1, 0]), int(angle_final_coord2d[1, 1])),
        #             (0, 255, 0),
        #             5,
        #         )
        #         img = cv2.circle(
        #             img,
        #             (int(angle_final_coord2d[1, 0]), int(angle_final_coord2d[1, 1])),
        #             8,
        #             (0, 255, 0),
        #             5,
        #         )

        #         # draw aim point
                
        #         aim_coord3d = np.concatenate([
        #             aim_point[None],
        #             np.ones((1, 1), dtype=np.float32) * -1.8,
        #             np.ones((1, 1), dtype=np.float32)], axis=1)
                
        #         aim_coord2d = (self.lidar2img[cam] @ aim_coord3d.T).T
        #         aim_coord2d = aim_coord2d[aim_coord2d[:, 2] > 1e-5]
        #         if len(aim_coord2d) > 0:
        #             aim_coord2d = aim_coord2d[:, :2] / aim_coord2d[:, 2:3]
        #             img = cv2.circle(
        #                 img, (int(aim_coord2d[0, 0]), int(aim_coord2d[0, 1])), 10, (0, 255, 255), 8,)
                
        #         # draw aim_dist circle
        #         # aim_dist를 원의 반지름으로 하는 원을 그려야해
        #         # 원의 테두리만 그려야해 선 색은 빨간색이고 선두께는 2, 원 안은 투명
        #         if aim_dist > 0:
        #             circle_angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False, dtype=np.float32)
        #             circle_xy = np.stack([
        #                 np.cos(circle_angles) * aim_dist,
        #                 np.sin(circle_angles) * aim_dist,
        #             ], axis=1)
        #             circle_coord3d = np.concatenate([
        #                 circle_xy,
        #                 np.ones((len(circle_xy), 1), dtype=np.float32) * -1.8,
        #                 np.ones((len(circle_xy), 1), dtype=np.float32),
        #             ], axis=1)
        #             circle_coord2d = (self.lidar2img[cam] @ circle_coord3d.T).T
        #             circle_valid = circle_coord2d[:, 2] > 1e-5

        #             projected_circle = np.zeros((len(circle_xy), 2), dtype=np.float32)
        #             projected_circle[circle_valid] = (
        #                 circle_coord2d[circle_valid, :2] / circle_coord2d[circle_valid, 2:3]
        #             )

        #             circle_segments = []
        #             seg_start = None
        #             for idx, is_valid in enumerate(circle_valid):
        #                 if is_valid and seg_start is None:
        #                     seg_start = idx
        #                 elif not is_valid and seg_start is not None:
        #                     if idx - seg_start >= 2:
        #                         circle_segments.append(projected_circle[seg_start:idx])
        #                     seg_start = None

        #             if seg_start is not None:
        #                 if len(circle_valid) - seg_start >= 2:
        #                     circle_segments.append(projected_circle[seg_start:])

        #             if len(circle_segments) >= 2 and circle_valid[0] and circle_valid[-1]:
        #                 merged_segment = np.vstack([circle_segments[-1], circle_segments[0]])
        #                 circle_segments = [merged_segment] + circle_segments[1:-1]

        #             for segment in circle_segments:
        #                 img = cv2.polylines(
        #                     img,
        #                     [np.round(segment).astype(np.int32)],
        #                     isClosed=False,
        #                     color=(255, 0, 0),
        #                     thickness=2,
        #                 )

        #         # # draw best_norm circle
        #         if best_norm > 0:
        #             circle_angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False, dtype=np.float32)
        #             circle_xy = np.stack([
        #                 np.cos(circle_angles) * best_norm,
        #                 np.sin(circle_angles) * best_norm,
        #             ], axis=1)
        #             circle_coord3d = np.concatenate([
        #                 circle_xy,
        #                 np.ones((len(circle_xy), 1), dtype=np.float32) * -1.8,
        #                 np.ones((len(circle_xy), 1), dtype=np.float32),
        #             ], axis=1)
        #             circle_coord2d = (self.lidar2img[cam] @ circle_coord3d.T).T
        #             circle_valid = circle_coord2d[:, 2] > 1e-5

        #             projected_circle = np.zeros((len(circle_xy), 2), dtype=np.float32)
        #             projected_circle[circle_valid] = (
        #                 circle_coord2d[circle_valid, :2] / circle_coord2d[circle_valid, 2:3]
        #             )

        #             circle_segments = []
        #             seg_start = None
        #             for idx, is_valid in enumerate(circle_valid):
        #                 if is_valid and seg_start is None:
        #                     seg_start = idx
        #                 elif not is_valid and seg_start is not None:
        #                     if idx - seg_start >= 2:
        #                         circle_segments.append(projected_circle[seg_start:idx])
        #                     seg_start = None

        #             if seg_start is not None:
        #                 if len(circle_valid) - seg_start >= 2:
        #                     circle_segments.append(projected_circle[seg_start:])

        #             if len(circle_segments) >= 2 and circle_valid[0] and circle_valid[-1]:
        #                 merged_segment = np.vstack([circle_segments[-1], circle_segments[0]])
        #                 circle_segments = [merged_segment] + circle_segments[1:-1]

        #             for segment in circle_segments:
        #                 img = cv2.polylines(
        #                     img,
        #                     [np.round(segment).astype(np.int32)],
        #                     isClosed=False,
        #                     color=(0, 128, 128),
        #                     thickness=2,
        #                 )


        #     # draw target point on image
        #     if cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        #         cmd_coord3d = np.concatenate([target_point,
        #                                       np.ones_like(target_point[:1]) * -1.8,
        #                                       np.ones_like(target_point[:1])], axis=0)
        #         cmd_coord2d = (self.lidar2img[cam] @ cmd_coord3d.T).T
        #         cmd_coord2d = cmd_coord2d[:2] / np.abs(cmd_coord2d[2:3])
        #         img = cv2.circle(img, (int(cmd_coord2d[0]), int(cmd_coord2d[1])), 7, (255, 105, 120), 6)

        #     # text and resize
        #     img = cv2.putText(img, cam, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        #     img = cv2.resize(img, (rw, rh))
        #     imgs.append(img)

        ### plot bev
        bev_img = tick_data['bev']
        bev_dict = self.sensors()[-1]
        if bev_dict['id']=='bev':
            ## agent
            # draw agent box on bev
            if 'boxes_3d' in output_batch[0]['img_bbox']:
                pred_bboxes3d, pred_labels3d, pred_trajs3d = self.get_bboxes(output_batch)
                for idx, (pred_bbox3d, pred_label3d) in enumerate(zip(pred_bboxes3d, pred_labels3d)):
                    box = self.convert_bev_bbox(pred_bbox3d, bev_dict)
                    cv2.polylines(bev_img, [box], isClosed=True, color=(0, 0, 255), thickness=1)

                    # draw agent motion traj on bev
                    if pred_trajs3d is not None and pred_label3d in [0, 1, 2, 3]:
                        pred_traj = np.concatenate([pred_bbox3d[:2][None], pred_trajs3d[idx]])
                        bev_coord = self.convert_bev_coord(pred_traj, bev_dict)
                        bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(0, 200, 0), line_thickness=1,
                                                       point_color=(0, 200, 0), point_thickness=1, point_radius=2)

            ### ego
            # draw ego box on bev
            box = self.convert_bev_bbox((0, 0, 0, 1.84, 4.89, 1.49, 0), bev_dict)
            cv2.polylines(bev_img, [box], isClosed=True, color=(255, 255, 0), thickness=1)

            # draw bev aim_dist circle
            if aim_dist > 0:
                bev_cam_z = bev_dict['z']
                bev_cam_fov = bev_dict['fov']
                bev_width = bev_dict['width']
                bev_height = bev_dict['height']
                bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2
                bev_ratio_w = bev_width / bev_range
                bev_ratio_h = bev_height / bev_range
                bev_center = self.convert_bev_coord(np.array([0.0, 0.0], dtype=np.float32), bev_dict)
                bev_img = cv2.ellipse(
                    bev_img,
                    (int(bev_center[0]), int(bev_center[1])),
                    (
                        max(1, int(round(aim_dist * bev_ratio_w))),
                        max(1, int(round(aim_dist * bev_ratio_h))),
                    ),
                    0,
                    0,
                    360,
                    (255, 0, 0),
                    2,
                )
            # draw bev best_norm circle
            if best_norm > 0:
                bev_cam_z = bev_dict['z']
                bev_cam_fov = bev_dict['fov']
                bev_width = bev_dict['width']
                bev_height = bev_dict['height']
                bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2
                bev_ratio_w = bev_width / bev_range
                bev_ratio_h = bev_height / bev_range
                bev_center = self.convert_bev_coord(np.array([0.0, 0.0], dtype=np.float32), bev_dict)
                bev_img = cv2.ellipse(
                    bev_img,
                    (int(bev_center[0]), int(bev_center[1])),
                    (
                        max(1, int(round(best_norm * bev_ratio_w))),
                        max(1, int(round(best_norm * bev_ratio_h))),
                    ),
                    0,
                    0,
                    360,
                    (0, 128, 128),
                    2,
                )

            # draw ego spatial traj
            if with_spatial_planning:
                bev_coord = self.convert_bev_coord(spatial_planning, bev_dict)
                bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(0, 200, 255), line_thickness=1,
                                               point_color=(0, 200, 255), point_thickness=1, point_radius=2)


            # draw bev steer point and line
            bev_coord = self.convert_bev_coord(np.array([[0.0, 0.0], steer_point], dtype=np.float32), bev_dict)
            bev_img = cv2.line(
                bev_img,
                (int(bev_coord[0, 0]), int(bev_coord[0, 1])),
                (int(bev_coord[1, 0]), int(bev_coord[1, 1])),
                (255, 165, 0),
                4,
            )
            # cv2.circle(bev_img, (int(bev_coord[1, 0]), int(bev_coord[1, 1])), 5, (0, 165, 255), 3)
            
            
            
            # draw ego temporal traj
            bev_coord = self.convert_bev_coord(pred_planning, bev_dict)
            bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(255, 0, 0), line_thickness=2,
                                           point_color=(255, 0, 0), point_thickness=6, point_radius=8)
            
            
            
            bev_coord = self.convert_bev_coord(np.array([[0.0, 0.0], angle_final_point], dtype=np.float32), bev_dict)
            bev_img = cv2.line(
                bev_img,
                (int(bev_coord[0, 0]), int(bev_coord[0, 1])),
                (int(bev_coord[1, 0]), int(bev_coord[1, 1])),
                (0, 255, 0),
                1,
            )
            # cv2.circle(bev_img, (int(bev_coord[1, 0]), int(bev_coord[1, 1])), 4, (255, 190, 255), 2)
            


            # draw bev aim point
            bev_coord = self.convert_bev_coord(aim_point, bev_dict)
            cv2.circle(bev_img, (int(bev_coord[0]), int(bev_coord[1])), 5, (0, 255, 255), 5)

            # draw bev target point
            bev_coord = self.convert_bev_coord(target_point, bev_dict)
            cv2.circle(bev_img, (int(bev_coord[0]), int(bev_coord[1])), 6, (255, 0, 120), 4)

        cmd_str = str(tick_data['command']).split('.')[-1]
        bev_panel_width = 460
        bev_canvas = np.zeros(
            (bev_img.shape[0], bev_img.shape[1] + bev_panel_width, bev_img.shape[2]),
            dtype=bev_img.dtype,
        )
        bev_canvas[:, bev_panel_width:] = bev_img
        panel_lines = [f"cmd: {cmd_str}"]
        panel_lines.extend(format_panel_lines("aim_point", aim_point))
        panel_lines.extend(format_panel_lines("aim_dist", aim_dist))
        panel_lines.extend(format_panel_lines("pred_temp_traj", pred_temp_traj))
        panel_lines.extend(format_panel_lines("wrong_traj", wrong_traj))

        text_y = 40
        for idx, line in enumerate(panel_lines):
            font_scale = 0.85 if idx == 0 else 0.55
            thickness = 2 if idx == 0 else 1
            bev_canvas = cv2.putText(
                bev_canvas,
                line,
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            text_y += 36 if idx == 0 else 26
        bev_img = bev_canvas

        # text and resize
        # bev_img = cv2.putText(bev_img, "cmd:{}".format(self.pid_metadata['command']), (15, bev_dict['height']-95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "speed:{:.2f}".format(self.pid_metadata['speed']), (15, bev_dict['height']-75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "steer:{:.2f}".format(self.pid_metadata['steer']), (15, bev_dict['height']-55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "throttle:{:.2f}".format(self.pid_metadata['throttle']), (15, bev_dict['height']-35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "brake:{:.2f}".format(self.pid_metadata['brake']), (15, bev_dict['height']-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, str(tick_data['command']), (15, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        frame = self.step
        Image.fromarray(bev_img).save(self.save_path / 'images' / ('%04d.jpg' % frame))
        bev_img = cv2.resize(bev_img, (rh*2, rh*2))

        # # merge
        # front = imgs[0]
        # front_left = imgs[1]
        # front_right = imgs[2]
        # back = imgs[3]
        # back_left = imgs[4]
        # back_right = imgs[5]

        # line1 = np.hstack([front_left, front, front_right])
        # line2 = np.hstack([back_right, back, back_left])
        # merge_img = np.vstack([line1, line2])
        # merge_img = np.hstack([merge_img, bev_img])

        #frame = self.step

        # Image.fromarray(bev_img).save(self.save_path / 'images' / ('%04d.jpg' % frame))

        outfile = open(self.save_path / 'metas' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def get_bboxes(self, output_batch):
        pred_bboxes3d = copy.deepcopy(output_batch[0]['img_bbox']['boxes_3d'].cpu().numpy())
        pred_scores3d = copy.deepcopy(output_batch[0]['img_bbox']['scores_3d'].cpu().numpy())
        pred_labels3d = copy.deepcopy(output_batch[0]['img_bbox']['labels_3d'].cpu().numpy())

        pred_mask3d = pred_scores3d > 0.3
        pred_bboxes3d = pred_bboxes3d[pred_mask3d]
        pred_labels3d = pred_labels3d[pred_mask3d]

        pred_trajs3d = None
        if 'trajs_3d' in output_batch[0]['img_bbox']:
            pred_trajs3d = copy.deepcopy(output_batch[0]['img_bbox']['trajs_3d'].cpu().numpy())
            pred_trajs_score = copy.deepcopy(output_batch[0]['img_bbox']['trajs_score'].cpu().numpy())

            pred_trajs3d = pred_trajs3d[pred_mask3d]
            pred_trajs_score = pred_trajs_score[pred_mask3d]

            # select top-one
            pred_trajs3d = pred_trajs3d[np.arange(len(pred_trajs3d)), pred_trajs_score.argmax(-1)]

        return pred_bboxes3d, pred_labels3d, pred_trajs3d

    def draw_trajectory(self, image, coord2d, line_color=(255, 0, 0), line_thickness=2,
                        point_color=(255, 0, 0), point_thickness=2, point_radius=None):

        for i in range(len(coord2d) - 1):
            point_start = coord2d[i]
            point_end = coord2d[i + 1]

            point_start = (int(point_start[0]), int(point_start[1]))
            point_end = (int(point_end[0]), int(point_end[1]))

            cv2.line(image, point_start, point_end, line_color, line_thickness)

            # draw points
            if point_radius is not None:
                cv2.circle(image, point_start, point_radius, point_color, point_thickness)
                cv2.circle(image, point_end, point_radius, point_color, point_thickness)

        return image

    def convert_bev_bbox(self, bbox, bev_dict):
        # box = self.convert_bev_bbox((0, 0, 0, 1.84, 4.89, 1.49, 0), bev_dict)
        bev_cam_z = bev_dict['z']
        bev_cam_fov = bev_dict['fov']
        bev_width = bev_dict['width']
        bev_height = bev_dict['height']

        bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2

        bev_ratio_w = bev_width / bev_range
        bev_ratio_h = bev_height / bev_range

        center = (bev_width / 2 + bbox[0] * bev_ratio_w,
                  bev_height / 2 - bbox[1] * bev_ratio_h)
        size = (bbox[3] * bev_ratio_w, bbox[4] * bev_ratio_h)
        angle = - bbox[6] / np.pi * 180

        rect = (center, size, angle)
        bev_bbox = cv2.boxPoints(rect)
        bev_bbox = np.int0(bev_bbox)

        return bev_bbox

    def convert_bev_coord(self, coord3d, bev_dict):
        bev_cam_z = bev_dict['z']
        bev_cam_fov = bev_dict['fov']
        bev_width = bev_dict['width']
        bev_height = bev_dict['height']

        bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2

        bev_ratio_w = bev_width / bev_range
        bev_ratio_h = bev_height / bev_range

        if len(coord3d.shape) == 1:
            coord_x = coord3d[0:1] * bev_ratio_w
            coord_y = coord3d[1:2] * bev_ratio_h
        else:
            coord_x = coord3d[:, 0:1] * bev_ratio_w
            coord_y = coord3d[:, 1:2] * bev_ratio_h

        offset_u = bev_width / 2 + coord_x
        offset_v = bev_height / 2 - coord_y
        coord2d = np.concatenate([offset_u, offset_v], axis=-1)

        return coord2d
