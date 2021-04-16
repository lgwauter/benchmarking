import time
import requests
from rvai.types import Inputs, Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image as Im
from rvai3client import RVAI3Client


API_KEY = 'd5de9c30-7660-4afc-a0d9-5700dd0ce366'
ENDPOINT = 'https://fairmot-test.azure-playground2.robovision.ai/pipelines/ray-worker-f964d/7008'
TEST_LENGTH = 1000
SESSION = requests.Session()
base_url = "https://fairmot-test.azure-playground2.robovision.ai/"
account = "admin"
password = "@dm1n"
minio1_url = "https://minio1.fairmot-test.azure-playground2.robovision.ai"
minio1_access_key = "robovision"
minio1_secret_key = "robovision"
minio2_url = "https://minio2.fairmot-test.azure-playground2.robovision.ai"
minio2_access_key = "robovision"
minio2_secret_key = "robovision"

client = RVAI3Client(
    rvai_base_url=base_url,
    rvai_account=account,
    rvai_password=password,
    minio1_server_url=minio1_url,
    minio1_access_key=minio1_access_key,
    minio1_secret_key=minio1_secret_key,
    minio2_server_url=minio2_url,
    minio2_access_key=minio2_access_key,
    minio2_secret_key=minio2_secret_key
)


# test_img.jpg is a production example of lux18 (1536x1536x3)
# it is important to take a full-size example
REQUEST_DATA = Inputs(image=Image(Im.open('test_img.jpg'))).to_json_struct()


def test_inference():
    """
    Send predict POST request.
    """
    # The HTTP headers we need for the API to understand us and let us in
    headers = {"Content-Type": "application/json", "api-key": API_KEY}

    url = f"{ENDPOINT}/api/v1/predict"
    # Actually send an HTTP POST request
    response = SESSION.post(
        url=url,
        json=REQUEST_DATA,
        headers=headers,
        timeout=60 * 3
    )
    print(url)
    return response.json()


if __name__ == '__main__':

    # send some frames first, since first frames are slower
    for i in range(5):
        test_inference()

    start = time.time()
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for i in range(TEST_LENGTH):
            future = executor.submit(
                test_inference,
            )
            futures.append(future)

        # with is blocking the Thread till all tasks are done
        # so this isn't really needed
        # for fut in as_completed(futures):
        #     print(fut.result())

    end = time.time()
    FPS = TEST_LENGTH / (end - start)
    print(f"FPS: {FPS}")
