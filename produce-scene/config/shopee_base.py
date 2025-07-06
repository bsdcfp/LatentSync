# 短时有效biz, 用于数据标注等可视化场景
upload_config_4249 = {
    'cid': 'sg',
    'env': 'live',
    'biz': 4249,
    'secret': '035ca5abd3b8e42833f402e9b4dd85f8'
}


# 永久有效biz, 用于投放和模版库等正式场景
upload_config_4253 = {
    'cid': 'sg',
    'env': 'live',
    'biz': 4253,
    'secret': '4963a23c02487519007b3f47f8440a3b'
}


# black image
black_image_id = 'sg-11134253-7rfhz-m3dackazk6xz4c'


register_upload_config = {
    4249: upload_config_4249,
    4253: upload_config_4253
}

catID2cat = {
    100016: "Women Bags",
    100532: "Women Shoes",
    100630: "Beauty",
    100013: "Mobile & Gadgets",
    100644: "Computers & Accessories",
    100017: "Women Clothes",
    100009: "Fashion Accessories",
    100637: "Sports & Outdoors",
    100629: "Food & Beverages",
    100534: "Watches",
    100633: "Baby & Kids Fashion",
    100641: "Motorcycles",
    100014: "Muslim Fashion",
    100011: "Men Clothes",
    100632: "Mom & Baby",
    100639: "Hobbies & Collections",
    100642: "Tickets, Vouchers & Services",
    100636: "Home & Living",
    100643: "Books & Magazines",
    100010: "Home Appliances",
    100001: "Health",
    100638: "Stationery",
    100533: "Men Bags",
    100012: "Men Shoes",
    100640: "Automobiles",
    100535: "Audio",
    100015: "Travel & Luggage",
    100635: "Cameras & Drones",
    100634: "Gaming & Consoles",
    100631: "Pets",
    100531: "Food Delivery",
    102053: "Deals Near Me"
}

def value_to_key_mapping(categories):
    """根据值获取键的映射"""
    return {value: key for key, value in categories.items()}

l12id = value_to_key_mapping(catID2cat)