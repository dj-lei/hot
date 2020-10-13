import re
import time
import pandas as pd
import configparser
import argparse
from spider.spider import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--firefox_profile", type=str, default="/home/ml/.mozilla/firefox/0k5sh6ex.default", help="firefox driver addr")
    parser.add_argument("--crawl_config", type=str, default="config/crawl.cfg", help="spider config")
    opt = parser.parse_args()

    cf = configparser.ConfigParser()
    cf.read(opt.crawl_config)

    # Is there any key words in redis yet

    # First crawl key words of popular trend
    key_words = []
    spider_kw = KeyWordsSpider(opt.firefox_profile)
    for platform, url in cf.items('url'):
        cfg = cf.options(platform + '_parse_theme')
        key_words.extend(spider_kw.run(url, cfg))
    spider_kw.close()

    # Secend crawl the corresponding key word of platform of best seller
    best_seller_data = pd.DataFrame()
    spider_sing_page = SinglePageSpider(opt.firefox_profile)
    for platform, url in cf.items('best_seller_search_url'):
        cfg = cf.items(platform + '_parse_links')
        cfg.pop(0)
        cfg.pop(0)
        best_seller_data = best_seller_data.append(spider_sing_page.run(url, cfg)).reset_index(drop=True)
    spider_sing_page.close()

    # Third crawl the corresponding key word of platform of source goods
    source_goods = pd.DataFrame()
    spider_multi_pages = MultiPageSpider(opt.firefox_profile)
    for platform, url in cf.items('source_goods_url'):
        cfg = cf.items(platform + '_parse_links')
        source_goods = source_goods.append(spider_multi_pages.run(url, cfg)).reset_index(drop=True)
    spider_multi_pages.close()

