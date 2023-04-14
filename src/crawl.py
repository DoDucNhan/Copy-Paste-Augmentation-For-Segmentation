import os
import cv2
import json
import argparse
from datetime import date
from utils import image_resize
from icrawler.builtin import FlickrImageCrawler, GoogleImageCrawler


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default='crawl_images',
    help="path to the download directory")
parser.add_argument("-n", "--num", type=int, default=100,
	help="number of download images")
parser.add_argument("-p", "--platform", type=str, default='google',
	help="cloud platform to crawl (google/flickr)")
parser.add_argument("-k", "--keywords", type=str, required=True,
	help="multiple keywords to crawl images seperated by comma")

args = vars(parser.parse_args())


def get_api_key():
    with open("files/flickr_key.json") as f:
        keys = json.load(f)
        return keys['key']


def get_keywords():
    with open("files/keywords.json") as f:
        return json.load(f)


if __name__ == '__main__':
    # start crawling
    keywords = get_keywords()
    api_key = get_api_key()
    directories = []
    keyword_list = args['keywords'].split(',')
    crawl_option = [keyword.strip() for keyword in keyword_list]
    print("---------------------Start crawling--------------------------")
    for keyword in crawl_option:
        img_dir = os.path.join(args['dir'], keyword)
        directories.append(img_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        if args['platform'] == 'google':
            google_crawler = GoogleImageCrawler(storage={'root_dir': img_dir})
            google_crawler.crawl(keyword=keywords[keyword], max_num=args['num'], min_size=(500,500))  
        else:
            flickr_crawler = FlickrImageCrawler(apikey=api_key,
                                        storage={'root_dir': img_dir})
            flickr_crawler.crawl(max_num=args['num'], tags=keywords[keyword], min_upload_date=date(2015, 5, 1))

    print("---------------------Finish crawling-------------------------")

    # start preprocessing - resize image to standard size
    standard_size = 720
    print("---------------------Start preprocessing--------------------------")
    for directory in directories:
    # for keyword in args['keywords']:
    #     directory = os.path.join(args['dir'], keyword)
        for filename in os.listdir(directory):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img.shape[0] == img.shape[1]:
                width = height = standard_size
            elif img.shape[0] > img.shape[1]:
                width, height = None, standard_size
            else:
                width, height = standard_size, None

            img = image_resize(img, width, height)
            # img = image_resize(img, standard_size, standard_size)
            cv2.imwrite(img_path, img)

    print("---------------------Finish preprocessing-------------------------")

