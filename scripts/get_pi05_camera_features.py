from pprint import pprint
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

policy = PI05Policy.from_pretrained("lerobot/pi05_base")
pprint(policy.config)
