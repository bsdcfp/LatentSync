######################################################################
#
# Copyright (c) 2024 Shopee Inc. All Rights Reserved.
#
######################################################################

"""
file: locust_press.py
author: min.yang@shopee.com
date: 2024-04-16 10:00:00
brief: ---
"""

import json
import glob
import random
import base64
import os
import json

from locust import HttpUser, task, FastHttpUser

images_root = "./examples/test_data"

def build_request(batch_size = 1):

    # url = 'https://http-gateway.spex.shopee.sg/sprpc/ais.spexmaas.infer'
    url = '/sprpc/ais.spexmaas.infer'
    headers = {
        'shopee-baggage': 'AIS_SPEX=109835',
        'Content-Type': 'application/json',
        'x-sp-sdu': 'aip.spextest.sg.live.master.default',
        'x-sp-servicekey': '19f451b01d3c3c391f210401e20a6520',
        'x-sp-timeout': '60000',
        'x-sp-processid': 'process_1'
    }
    data = {
        "data": {
            "entries_map": {
                "foreground_url_list": {
                    "type": "ValueMessage_ValStringList",
                    "ValueOneof": {
                        "ValStringList": {
                            "data": [
                                "https://down-br.img.susercontent.com/br-11134253-7r98o-ly1qmp3d40omde"
                            ]
                        }
                    }
                },
                "template_image_url_list": {
                    "type": "ValueMessage_ValStringList",
                    "ValueOneof": {
                        "ValStringList": {
                            "data": [
                                "https://down-br.img.susercontent.com/br-11134253-7r98o-ly1qmrjvlawma6"
                            ]
                        }
                    }
                },
                "template_config_list": {
                    "type": "ValueMessage_ValStringList",
                    "ValueOneof": {
                        "ValStringList": {
                            "data": [
                                "{\"text_prompt\":\"a foreground item palced on a wooden surface, empty surface, soft light, shadow\",\"extra_info\":{\"ip_adapter_scale\":0.6,\"target_h\":0.8,\"target_w\":0.8,\"target_center\":[0.5,0.5]}}"
                            ]
                        }
                    }
                }
            }
        }
    }


    return url, headers, data

class MyUser(HttpUser):

    @task
    def process(self):
        url,headers,req_data = build_request()
        with self.client.post("/api/process", headers=headers, data=json.dumps(req_data)) as res:
            if res.status_code != 200:
                print("Didn't detect bad response, got: " + str(res.status_code))
