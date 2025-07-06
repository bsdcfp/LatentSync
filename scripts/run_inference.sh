curl -X POST \
  -H 'shopee-baggage: AIS_SPEX=106082' \
  -H 'Content-Type: application/json' \
  -H 'x-sp-sdu: aip.spextest.sg.live.master.default' \
  -H 'x-sp-servicekey: 19f451b01d3c3c391f210401e20a6520' \
  -H 'x-sp-timeout: 60000' \
  -H 'x-sp-processid: process_1' \
  --data '{
    "data": {
      "entries_map": {
        "prompt_list": {
          "type": "ValueMessage_ValStringList",
          "ValueOneof": {
            "ValStringList": {
              "data": [
                "a (model:0.8) photoshoot, high quality, highly detailed, 4k, cinematic, indoor, men, "
              ]
            }
          }
        },
        "negative_prompt_list": {
          "type": "ValueMessage_ValStringList",
          "ValueOneof": {
            "ValStringList": {
              "data": [
                "nude, watermark, noisy, glitch, glitch, noise, jpeg artifacts, bad , dirty, ugly, Cluttered ground, (strong shadow), (strong light),"
              ]
            }
          }
        },
        "ip_image_list": {
          "type": "ValueMessage_ValStringList",
          "ValueOneof": {
            "ValStringList": {
              "data": [
                "https://down-br.img.susercontent.com/br-11134253-7r98o-lwmmsqi4tzli71"
              ]
            }
          }
        },
        "inpaint_input_list": {
          "type": "ValueMessage_ValStringList",
          "ValueOneof": {
            "ValStringList": {
              "data": [
                "https://down-br.img.susercontent.com/br-11134253-7r98o-lwmmu91r0h2udc"
              ]
            }
          }
        },
       "mask_eroded_num_list": {
          "type": "ValueMessage_ValInt32List",
          "ValueOneof": {
            "ValInt32List": {
              "data": [
                -4
              ]
            }
          }
        },
       "control_eroded_num_list": {
          "type": "ValueMessage_ValInt32List",
          "ValueOneof": {
            "ValInt32List": {
              "data": [
                -4
              ]
            }
          }
        },
       "seed_list": {
          "type": "ValueMessage_ValInt32List",
          "ValueOneof": {
            "ValInt32List": {
              "data": [
                20
              ]
            }
          }
        }
      }
    }
  }' \
  'https://http-gateway.spex.shopee.sg/sprpc/ais.spexmaas.infer' \
  -i