from typing import Dict
from afforddp.policy.base_pointcloud_policy import BasePointcloudPolicy


class BasePointcloudRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePointcloudPolicy) -> Dict:
        raise NotImplementedError()