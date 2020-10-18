import re
import time
import pandas as pd
import configparser
import argparse
from spider.spider import *
from db.db import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--firefox_profile", type=str, default="/home/ml/.mozilla/firefox/0k5sh6ex.default", help="firefox driver addr")
    parser.add_argument("--crawl_config", type=str, default="config/crawl.cfg", help="spider config")
    parser.add_argument("--redis_config", type=str, default="172.17.0.2:6379:1", help="redis config")
    parser.add_argument("--mysql_config", type=str, default="172.17.0.3:root:123456:hot", help="mysql config")
    opt = parser.parse_args()

    cf = configparser.ConfigParser()
    cf.read(opt.crawl_config)

    redis_ctrl = RedisCtrl(opt.redis_config.split(':'))
    mysql_ctrl = MysqlCtrl(opt.mysql_config.split(':'))

    key_words_number = redis_ctrl.get_key_number('key_words')
    # Is there any key words in redis yet
    if key_words_number == 0:
        # First crawl key words of popular trend
        key_words = []
        spider_kw = KeyWordsSpider(opt.firefox_profile)
        for platform, url in cf.items('url'):
            cfg = dict(cf.items(platform + '_parse_theme'))
            key_words.extend(spider_kw.run(url, cfg))
        spider_kw.close()

    for _ in range(0, key_words_number):
        # Secend crawl the corresponding key word of platform of best seller
        key_words = eval(redis_ctrl.redis_con.spop('key_words'))[1]
        best_seller_data = pd.DataFrame()
        spider_sing_page = SinglePageSpider(opt.firefox_profile)
        for platform, url in cf.items('best_seller_search_url'):
            url = url.replace(re.findall('&q=(.*?)&', url)[0], key_words)
            cfg = cf.items(platform + '_parse_links')
            cfg.pop(0)
            cfg.pop(0)
            best_seller_data = best_seller_data.append(spider_sing_page.run(url, dict(cfg))).reset_index(drop=True)
        spider_sing_page.close()

        # Third crawl the corresponding key word of platform of source goods
        source_goods = pd.DataFrame()
        spider_multi_pages = MultiPageSpider(opt.firefox_profile)
        for platform, url in cf.items('source_goods_url'):
            url = url.replace(re.findall('\?so=(.*?)&', url)[0], key_words)
            cfg = cf.items(platform + '_parse_links')
            source_goods = source_goods.append(spider_multi_pages.run(url, dict(cfg))).reset_index(drop=True)
        spider_multi_pages.close()

    mysql_ctrl.save(best_seller_data, 'best_seller')
    mysql_ctrl.save(source_goods, 'source_goods')
