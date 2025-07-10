######################################################################
#
# Copyright (c) 2022 Shopee Inc. All Rights Reserved.
#
######################################################################
"""
file: server.py
author: finch.li@shopee.com
date: 2022-05-09 10:30:41
brief: ---
"""

import json
import orjson
import socket
import logging

from flask import Flask, request, Response, stream_with_context  # type: ignore
from pydantic import ValidationError, BaseModel
from aipinfer import logger, exceptions
from typing import Callable, Optional, Type, Any
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from . import monitor
from . import jobs
from .params import BaseResParams, ReqParams, ResParams


WorkerType = Callable[[BaseModel, BaseResParams], None]


class AIPApp(Flask):
    def __init__(self, import_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(import_name, *args, **kwargs)
        self.setup()

        self.worker: Optional[WorkerType] = None
        self.req_params: Type[BaseModel] = ReqParams
        self.res_params: Type[BaseResParams] = ResParams

    def setup(self) -> None:
        logger.debug("AIPApp setup!")

        self.logger.disabled = True
        monitor.init()
        jobs.start()

    def register(self,
                 _worker: WorkerType,
                 _req_params: Optional[Type[BaseModel]] = None,
                 _res_params: Optional[Type[BaseResParams]] = None):
        if not callable(_worker):
            raise exceptions.BackendError("the registered worker is not callable")

        self.worker = _worker
        if _req_params is not None:
            self.req_params = _req_params
        if _res_params is not None:
            try:
                _res_params()
            except Exception as exc:  # pylint: disable=broad-except
                raise TypeError(
                    "Your self-defined ResParams cannot be initialized without parameters directly, which "
                    "is not supported by the service") from exc

            self.res_params = _res_params


app = AIPApp(__name__)
LOG = logging.getLogger("werkzeug")
LOG.disabled = True


def register(_worker: WorkerType,
             _req_params: Optional[Type[BaseModel]] = None,
             _res_params: Optional[Type[BaseResParams]] = None):
    logger.warn("Deprecated, register interface will be removed in the next version, please use app.register()")
    app.register(_worker, _req_params, _res_params)


@app.route("/api/process", methods=["POST"])
def process():
    with monitor.IN_PROGRESS_TRACKER:
        res_params = app.res_params()

        try:
            with monitor.MonitorTimer("getdata"):
                get_data = request.data
                logger.add_notice("datasize", len(get_data))

            with monitor.MonitorTimer("datadecode"):
                # req_params = app.req_params(**json.loads(get_data.decode()))
                # 智能解析：优先尝试 orjson，失败则回退到标准 json
                try:
                    # 1. 尝试用 orjson 直接解析 bytes。
                    #    这是最高性能的路径，专门为你发送 Numpy 数组的客户端准备。
                    payload_dict = orjson.loads(get_data)
                except orjson.JSONDecodeError:
                    # 2. 如果 orjson 失败（比如收到了一个不含特殊类型的标准JSON字符串），
                    #    则回退到标准库的 json.loads。
                    #    我们先将 bytes 解码为 utf-8 字符串，这是 json.loads 所需的。
                    try:
                        payload_dict = json.loads(get_data.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        # 3. 如果两种方式都失败了，说明请求体格式确实有问题。
                        #    这里可以记录一个更明确的日志。
                        logger.error(f"Failed to decode request body with both orjson and standard json. Error: {e}")
                        # 抛出异常，让外层的 try...except 块去处理标准错误响应。
                        raise exceptions.AppException(
                            exceptions.NUM_DECODE_ERROR, 
                            "Request body is not valid JSON or orjson."
                        )
                
                # 无论哪种方式成功，都用得到的 payload_dict 创建 Pydantic 模型
                req_params = app.req_params(**payload_dict)

            if hasattr(req_params, "appid"):
                monitor.NUM_REQUESTS.labels(appid=req_params.appid,
                                            mpid=monitor.PID).inc()

            with monitor.MonitorTimer("total"):
                app.worker(req_params, res_params)  # type: ignore
        except json.decoder.JSONDecodeError as err:
            logger.error(err)
            res_params.err_num = exceptions.NUM_DECODE_ERROR
            res_params.err_msg = exceptions.desc_of_num(
                exceptions.NUM_DECODE_ERROR)
        except ValidationError as err:
            logger.error(err.errors())
            res_params.err_num = exceptions.NUM_ILLEGAL_ARGS
            res_params.err_msg = json.dumps(err.errors())
        except exceptions.AppException as err:
            logger.error(err.reason)
            res_params.err_num, res_params.err_msg = err.num, err.reason
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(exc)
            res_params.err_num = exceptions.NUM_UNKNOWN
            res_params.err_msg = str(exc)

        try:
            res_json = res_params.model_dump_json()
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(exc)
            res_params = app.res_params()
            res_params.err_num = exceptions.NUM_INIT_ERROR
            res_params.err_msg = str(exc)
            res_json = res_params.model_dump_json()
        finally:
            logger.add_notice("err_num", res_params.err_num)
            logger.add_notice("err_msg", res_params.err_msg)
            logger.flush()

        if res_params.err_num != 0:
            monitor.INTERVAL_STATUS.labels(err_num=str(res_params.err_num),
                                           err_msg=res_params.err_msg,
                                           mpid=monitor.PID).inc()

        return res_json


@app.route("/api/stream_process", methods=["POST"])
def stream_process():
    with monitor.IN_PROGRESS_TRACKER:
        res_params = app.res_params()
        try:
            with monitor.MonitorTimer("getdata"):
                get_data = request.data
                logger.add_notice("datasize", len(get_data))

            with monitor.MonitorTimer("datadecode"):
                req_params = app.req_params(**json.loads(get_data.decode()))

            if hasattr(req_params, "appid"):
                monitor.NUM_REQUESTS.labels(appid=req_params.appid,
                                            mpid=monitor.PID).inc()
            generation = app.worker(req_params, res_params)  # type: ignore

            # Streaming case
            def stream_results():
                try:
                    while True:
                        next(generation)  # type: ignore
                        yield (json.dumps(res_params.model_dump(), ensure_ascii=False) + '\0').encode('utf-8')
                except StopIteration:
                    # exit normally
                    pass
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception(exc)
                    res_params.result = None  # type: ignore
                    res_params.err_num = exceptions.NUM_UNKNOWN
                    res_params.err_msg = str(exc)
                    yield (json.dumps(res_params.model_dump(), ensure_ascii=False) + '\0').encode('utf-8')
                finally:
                    logger.add_notice("err_num", res_params.err_num)
                    logger.add_notice("err_msg", res_params.err_msg)
                    logger.flush()

            with monitor.MonitorTimer("total"):
                if req_params.stream:  # type: ignore
                    return stream_with_context(stream_results())
                else:
                    for _ in generation:  # type: ignore
                        pass
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(exc)
            res_params.err_num = exceptions.NUM_UNKNOWN
            res_params.err_msg = str(exc)

        try:
            res_json = res_params.model_dump_json()
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(exc)
            res_params = app.res_params()
            res_params.err_num = exceptions.NUM_INIT_ERROR
            res_params.err_msg = str(exc)
            res_json = res_params.model_dump_json()
        finally:
            logger.add_notice("err_num", res_params.err_num)
            logger.add_notice("err_msg", res_params.err_msg)
            logger.flush()

        if res_params.err_num != 0:
            monitor.INTERVAL_STATUS.labels(err_num=str(res_params.err_num),
                                           err_msg=res_params.err_msg,
                                           mpid=monitor.PID).inc()

        return res_json


@app.route("/api/ping", methods=["GET"])
def ping():
    logger.notice(f"ping: {monitor.PID}")

    return "{}: {}".format(socket.gethostname(), "ping")


@app.route("/metrics")
def metrics():
    logger.notice(f"metrics: {monitor.PID}")
    try:
        data = monitor.get_metrics()
        return Response(data, mimetype=CONTENT_TYPE_LATEST)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(exc)
