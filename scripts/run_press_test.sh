#!/bin/bash

locust_file_path=test/locust_press.py
#locust_file_path=./locust_press_urls.py
user_number=2 # the number of Users to spawn
spawn_rate=6 # the spawn rate (number of users to start per second)

time=3m

url="http://0.0.0.0:8081"
url="http://127.0.0.1:80"
# url=http://sg10.aip.mlp.shopee.io/aip-svc-31/spu-brand-detection

locust -f ${locust_file_path} --headless -u ${user_number} -r ${spawn_rate} --run-time ${time}  --host=${url}
