import os
import os.path as osp
import time

from fastapi import FastAPI, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse

CONFIG_TMP_DIR = "./tmp"
app = FastAPI()


def make_response(status_code, **kwargs):
    data = {'code': status_code, 'timestamp': time.time()}
    data.update(**kwargs)
    json_compatible_data = jsonable_encoder(data)
    resp = JSONResponse(content=json_compatible_data, status_code=status_code)
    return resp


@app.get("/")
def root():
    return RedirectResponse(url='/docs')


@app.post("/v1/create_calibration_task")
async def create_calibration_task(file: bytes = UploadFile):
    return make_response(200, filename=file.filename)


def main():
    if not osp.exists(CONFIG_TMP_DIR):
        os.mkdir(CONFIG_TMP_DIR)


if __name__ == '__main__':
    main()
