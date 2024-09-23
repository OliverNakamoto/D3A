import os
import requests
from flickrapi import FlickrAPI
from urllib.parse import urlparse

# Replace these with your Flickr API key and secret
FLICKR_PUBLIC = 'c32548c1f8a515d9072deb7381e6bb30'
FLICKR_SECRET = 'YOUa82e93eee655910a'

# Create directories to store images
os.makedirs('uncurated_dataset', exist_ok=True)

# Initialize Flickr API
flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

def download_images(query, max_photos):
    photos = flickr.photos.search(
        text=query,
        per_page=max_photos,
        media='photos',
        sort='relevance',
        safe_search=1,
        content_type=1,
        extras='url_o,url_c,license'
    )

    count = 0
    for photo in photos['photos']['photo']:
        url = photo.get('url_o') or photo.get('url_c')
        if not url:
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            file_name = os.path.join('uncurated_dataset', f"{photo['id']}.jpg")
            with open(file_name, 'wb') as f:
                f.write(response.content)
            count += 1
            print(f"Downloaded {count}/{max_photos} images", end='\r')
            if count >= max_photos:
                break
        except Exception as e:
            print(f"Failed to download image {photo['id']}: {e}")

# Download 1000 images of pigs and piglets
download_images('pig', 100)
