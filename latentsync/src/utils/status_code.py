class PI_STATUS_CODE(): 
    # SUCCESS is used when the aigc image is generated and passed AutoQC. The rest of the process remains RUNNING, 
    # which is used for each module to transmit intermediate results.
    SUCCESS = 0
    RUNNING = 1

    # error
    MATCH_ERROR = 10001
    DOWNLOAD_FG_ERROR = 10002
    DOWNLOAD_TEMPLATE_ERROR = 10003
    UPLOAD_ERROR = 10004
    GEN_PROMPT_ERROR = 10005
    NSFW_ERROR = 10006
    GENERATION_ERROR = 10007  # 添加视频生成错误状态码

    # filter
    FILTER_CROP_EDGE = 10101

    # strategy
    INVALID_STRATEGY = 10201

    # retry
    RETRY_TIMES_EXCEEDED = 10301