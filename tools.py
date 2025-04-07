import time
import requests

outer_ip = None
def get_ip():
    global outer_ip
    while outer_ip is None:
        try:
            res = requests.get('https://myip.ipip.net/').text
            if '228' in res:
                outer_ip = '228'
            elif '229' in res:
                outer_ip = '229'
            break
        except Exception as e:
            print(e)
            time.sleep(2)
    print(f">> 当前IP：{outer_ip}")

get_ip()
