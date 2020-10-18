import requests
from io import BytesIO
from PIL import Image
import argparse
from db.db import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_config", type=str, default="172.17.0.2:6379:1", help="redis config")
    parser.add_argument("--mysql_config", type=str, default="172.17.0.3:root:123456:hot", help="mysql config")
    opt = parser.parse_args()

    redis_ctrl = RedisCtrl(opt.redis_config.split(':'))
    mysql_ctrl = MysqlCtrl(opt.mysql_config.split(':'))

    items = mysql_ctrl.load('best_seller')
    for i in range(0, len(items)):
        try:
            if 'http' not in items['images_url_attribute'][i]:
                src = 'https:'+items['images_url_attribute'][i]
            else:
                src = items['images_url_attribute'][i]
            dir_path = 'data/best_seller/'+items['id'][i]+'.jpg'
            print(src)
            pic = Image.open(BytesIO(requests.get(src).content))
            pic = pic.convert('RGB')
            w, h = pic.size
            pic = pic.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
            pic.save(dir_path)
        except requests.exceptions.ConnectionError:
            print('Sorrry,image cannot downloaded, url is error{}.'.format(src))

    items = mysql_ctrl.load('source_goods')
    for i in range(0, len(items)):
        try:
            if 'http' not in items['images_url_attribute'][i]:
                src = 'https:' + items['images_url_attribute'][i]
            else:
                src = items['images_url_attribute'][i]
            dir_path = 'data/source_goods/' + items['id'][i] + '.jpg'
            print(src)
            pic = Image.open(BytesIO(requests.get(src).content))
            pic = pic.convert('RGB')
            w, h = pic.size
            pic = pic.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
            pic.save(dir_path)
        except requests.exceptions.ConnectionError:
            print('Sorrry,image cannot downloaded, url is error{}.'.format(src))