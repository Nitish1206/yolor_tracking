import cv2
import requests

def get_result_api(image):
    api_token = '8e5bd302366d69bdb9b2672008000405705f4294'
    url = 'https://api.platerecognizer.com/v1/plate-reader'
    headers = { 'Authorization': 'Token {0}'.format(api_token)}
    cv2.imwrite("temp.jpg", image)
    files = {'upload': open('temp.jpg', 'rb')}
    values = {'regions': 'gb'}
    data = requests.post(url, files=files, data=values, headers=headers)
    result = data.json()
    # print(15*"*")
    # print(data.json())
    # print(15*"*")
    if result.get("results"):
        plate = result['results'][0]['plate']
        return plate
    else:
        return None
        
if __name__ == "__main__":
    image_path=r"N:\Projects\yolor_tracking\videos\images\vlcsnap-2022-04-05-18h36m45s585 (3).png"
    image=cv2.imread(image_path)
    plate_number=get_result_api(image)
    print(plate_number)