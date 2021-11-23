import os
from minio import Minio
from typing import List
import asyncio
import logging
import json

logging.basicConfig(level=logging.INFO)

OSS_ENDPOINT = os.environ['OSS_ENDPOINT']
OSS_DST_BUCKET = os.environ['OSS_DST_BUCKET']
OSS_ACCESSKEY = os.environ['OSS_ACCESSKEY']
OSS_SECRETKEY = os.environ['OSS_SECRETKEY']
DOWNLOAD_DIR = './measurements'


async def download(object_name: str):
    filename = os.path.join(DOWNLOAD_DIR, f'{object_name}')
    minioClient = Minio(OSS_ENDPOINT, access_key=OSS_ACCESSKEY, secret_key=OSS_SECRETKEY, secure=False)  # Create MinIO client
    try:
        minioClient.fget_object(OSS_DST_BUCKET, object_name, filename)
        return {'status': True, 'file': filename}
    except Exception as err:
        return {'status': False, 'error': str(err)}


async def main():
    # Init minio client
    minioClient = Minio(OSS_ENDPOINT, access_key=OSS_ACCESSKEY, secret_key=OSS_SECRETKEY, secure=False)

    objects_list = list(map(lambda x: x.object_name, minioClient.list_objects(OSS_DST_BUCKET)))
    if len(objects_list) > 0:
        logging.info(f"Got {len(objects_list)} measurements")

    # Filter out all failed_objects
    results_list = await asyncio.gather(*[download(object_name) for object_name in objects_list])

    print(json.dumps(results_list, indent=True))

if __name__ == '__main__':
    asyncio.run(main())
