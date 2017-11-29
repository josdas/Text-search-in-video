import datetime

import imageio
from bot.backend.queryhandler import QueryHandler

from backend.bothandler import BotHandler

token = ''
STAR_TOKEN = ['/s', '/search']
MAIN_DIR = 'data/'
MAX_QUERY_PER_MINUTE = 10


def main():
    bot = BotHandler(token)
    print('BotHandler created')

    now = datetime.datetime.now()
    text_handler = QueryHandler(MAIN_DIR + 'video_data',
                                MAIN_DIR + 'CNN.net',
                                MAIN_DIR + 'Africa.mp4')
    print('QueryHandler created')

    new_offset = None
    minute = now.minute
    delay = {}

    while True:
        bot.get_updates(new_offset)

        last_update = bot.get_last_update()

        if last_update:
            last_update_id = last_update['update_id']
            text = last_update['message']['text']
            chat_id = last_update['message']['chat']['id']

            if chat_id not in delay:
                delay[chat_id] = []

            delay_d = delay[chat_id]
            print('from {}:"{}".'.format(chat_id, text))

            if len(delay_d) < MAX_QUERY_PER_MINUTE or delay_d[-MAX_QUERY_PER_MINUTE] < minute:
                good = False
                for start in STAR_TOKEN:
                    if text[:len(start)] == start:
                        good = True
                        description = text[len(start):]
                if good:
                    delay[chat_id].append(minute)
                    try:
                        result = text_handler.get_link(description)
                    except:
                        bot.send_message(chat_id, 'Я не понял.')
                    else:
                        link = result['link']
                        img = result['img']
                        img_name = 'temp/{}.jpg'.format(result['id'])

                        imageio.imwrite(img_name, img)
                        bot.send_photo(chat_id, img_name)

                        # text_handler.plot_time_similarity(description)

                        print('{} for "{}"'.format(link, description))
                        bot.send_message(chat_id,
                                         'Ваш результат для фразы "{}": {}'.format(description, link))
                else:
                    bot.send_message(chat_id,
                                     'Введите "/search <text>" или "/s <text>" для поиска.')
            print(3 * '\n')
            new_offset = last_update_id + 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
