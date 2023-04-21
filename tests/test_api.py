from articulated_kit.client.realsense_recorder.api.default import status_v1_status_get
from articulated_kit.client.realsense_recorder import Client
client = Client(base_url="http://localhost:5050")
print(status_v1_status_get.sync_detailed(client=client))