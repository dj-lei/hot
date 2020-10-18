from selenium import webdriver
from abc import abstractmethod
import pandas as pd
import time
import uuid
import re
from selenium.webdriver.common.action_chains import ActionChains


def match_chinese(string):
    return re.findall('[\u4e00-\u9fa5]+', string)[0]


class Spider(object):
    def __init__(self, profile=None):
        self.profile = webdriver.FirefoxProfile(profile)
        self.driver = webdriver.Firefox(profile)
        self.cf = ''

    @abstractmethod
    def run(self, url, cf):
        pass

    def close(self):
        self.driver.quit()


class KeyWordsSpider(Spider):
    def __init__(self, profile):
        super(KeyWordsSpider, self).__init__(profile)

    def run(self, url, cf):
        self.cf = cf

        self.driver.get(url)
        data = []

        while len(self.driver.find_elements_by_xpath(cf.get('move_to_element'))) == 0:
            pass
        ele = self.driver.find_elements_by_xpath(cf.get('move_to_element'))[0]
        ActionChains(self.driver).move_to_element(ele).perform()

        while len(self.driver.find_elements_by_xpath(cf.get('themes'))) == 0:
            ActionChains(self.driver).move_to_element(ele).perform()

        themes = self.driver.find_elements_by_xpath(cf.get('themes'))
        for i, theme in enumerate(themes):
            theme_text = match_chinese(theme.get_attribute('textContent'))
            son_parse = cf.get('son_themes').split('+str(i+1)+')
            son_themes_text = self.driver.find_elements_by_xpath(son_parse[0] + str(i + 1) + son_parse[1])
            for j, son_theme in enumerate(son_themes_text):
                son_theme_text = match_chinese(son_theme.get_attribute('textContent'))
                data.append([theme_text, son_theme_text])
        return data


class SinglePageSpider(Spider):
    def __init__(self, profile):
        super(SinglePageSpider, self).__init__(profile)

    def run(self, url, cf):
        self.cf = cf
        items_list = list(self.cf.keys())

        self.driver.get(url)
        data = pd.DataFrame([], columns=items_list)
        # wait info load
        for i in range(0, 40):
            self.driver.execute_script("document.documentElement.scrollTop=" + str(i * 100 + 100))
            time.sleep(0.1)

        # extraction info
        while True:
            try:
                for item in items_list:
                    if 'attribute' in item:
                        temp = self.cf.get(item).split('|')
                        a = self.driver.find_elements_by_xpath(temp[0])
                        data[item] = pd.Series([a[i].get_attribute(temp[1]) for i in range(0, len(a))])
                    else:
                        a = self.driver.find_elements_by_xpath(self.cf.get(item))
                        data[item] = pd.Series([a[i].text for i in range(0, len(a))])
                data['id'] = pd.Series([str(uuid.uuid4()) for i in range(0, len(a))])
            except:
                continue
            break

        return data


class MultiPageSpider(Spider):
    def __init__(self, profile):
        super(MultiPageSpider, self).__init__(profile)

    def run(self, url, cf):
        self.cf = cf
        current_url = url
        items_list = list(self.cf.keys())
        items_list.pop(0)  # pop page_num xpath
        items_list.pop(0)  # pop click xpath

        self.driver.get(url)
        page_nums = int(self.driver.find_elements_by_xpath(cf.get('page_num')[:-1])[0].text.split(
            cf.get('page_num')[-1])[1])
        page_nums = 5
        final = pd.DataFrame([], columns=items_list)
        data = pd.DataFrame([], columns=items_list)
        for page_num in range(0, page_nums):
            # wait info load
            while True:
                result = [len(self.driver.find_elements_by_xpath(cf.get(item))) for item in items_list]
                if (0 not in result) & (len(set(result)) == 1):
                    for i in range(0, 45):
                        self.driver.execute_script("document.documentElement.scrollTop=" + str(i * 200 + 200))
                        time.sleep(0.1)
                    time.sleep(0.5)
                    break

            # extraction info
            while True:
                try:
                    for item in items_list:
                        if 'attribute' in item:
                            temp = cf.get(item).split('|')
                            a = self.driver.find_elements_by_xpath(temp[0])
                            data[item] = pd.Series([a[i].get_attribute(temp[1]) for i in range(0, len(a))])
                        else:
                            a = self.driver.find_elements_by_xpath(cf.get(item))
                            data[item] = pd.Series([a[i].text for i in range(0, len(a))])
                    data['id'] = pd.Series([str(uuid.uuid4()) for i in range(0, len(a))])
                    data['page_num'] = page_num
                except:
                    continue
                break

            # quit when the per page of items of duplicate are greater than a half
            current_num = len(final)
            final = final.append(data).drop_duplicates(items_list).reset_index(drop=True)
            if len(final) < current_num + int(len(data) / 2):
                break

            # last page quit
            if page_num == (page_nums - 1):
                break

            # turn page,then waiting for refreshing of the page
            self.driver.find_elements_by_xpath(cf.get('click'))[0].click()
            while current_url == self.driver.current_url:
                pass

        return final
