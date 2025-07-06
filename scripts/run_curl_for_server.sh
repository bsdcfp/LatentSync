curl -X POST \
  -H 'shopee-baggage: AIS_SPEX=106285' \
  -H 'Content-Type: application/json' \
  -H 'x-sp-sdu: aip.spextest.sg.live.master.default' \
  -H 'x-sp-servicekey: 19f451b01d3c3c391f210401e20a6520' \
  -H 'x-sp-timeout: 60000' \
  -H 'x-sp-processid: process_1' \
  --data '{
    "data": {
      "entries_map": {
        "foreground_url_list": {
          "type": "ValueMessage_ValStringList",
          "ValueOneof": {
            "ValStringList": {
              "data": [
                "https://down-br.img.susercontent.com/br-11134253-7r98o-lwpidgzwwmmq97"
              ]
            }
          }
        },
        "template_image_url_list": {
          "type": "ValueMessage_ValStringList",
          "ValueOneof": {
            "ValStringList": {
              "data": [
                "https://down-br.img.susercontent.com/br-11134253-7r98o-lwpidjgpaovqcc"
              ]
            }
          }
        },
        "template_config_list": {
          "type": "ValueMessage_ValStringList",
          "ValueOneof": {
            "ValStringList": {
              "data": [
                "{\"text_prompt\": \"a foreground item palced on a wooden surface, empty surface, soft light, shadow\", \"extra_info\": {\"ip_adapter_scale\": 0.6, \"target_h\": 0.8, \"target_w\": 0.8, \"target_center\": [0.5, 0.5]}}"
              ]
            }
          }
        }
      }
    }
  }' \
  'https://http-gateway.spex.shopee.sg/sprpc/ais.spexmaas.infer' \
  -i