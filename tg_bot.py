import logging
import os
from dotenv import load_dotenv
import telebot
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from model_trainer import SmartCityGuideTrainer
from route_planner import SmartCityGuidePlanner
from telebot import apihelper



load_dotenv()
apihelper.SESSION_TIME_TO_LIVE = 5 * 60
apihelper.READ_TIMEOUT = 5
apihelper.CONNECT_TIMEOUT = 3

token = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(token)

# Глобальные переменные для хранения состояния пользователей
user_sessions = {}

# Структура для хранения данных пользователя
class UserSession:
    def __init__(self):
        self.current_state = 0
        self.user_interests = ""
        self.min_time = 2.0
        self.max_time = 4.0
        self.start_coords = None
        self.start_location = ""
        self.trained_model = None

def get_user_session(chat_id):
    if chat_id not in user_sessions:
        user_sessions[chat_id] = UserSession()
    return user_sessions[chat_id]

"""...................................................................."""

# Функция которая дефолтную клавиатуру делает
def create_default_keyboard():
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_start = telebot.types.KeyboardButton(text="Начало работы")
    button_help = telebot.types.KeyboardButton(text="Помощь")
    keyboard.add(button_start)
    keyboard.add(button_help)
    return keyboard

# Клавиатура на ввод
def create_address_input_keyboard():
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_location = telebot.types.KeyboardButton(text="Отправить геотег", request_location=True)
    button_back = telebot.types.KeyboardButton(text="Назад")
    keyboard.add(button_location)
    keyboard.add(button_back)
    return keyboard

# Клавиатура для ввода времени
def create_estimated_time_input_keyboard():
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_short = telebot.types.KeyboardButton(text="1-2 часа")
    button_medium = telebot.types.KeyboardButton(text="2-3 часа")
    button_long = telebot.types.KeyboardButton(text="3-4 часа")
    button_longest = telebot.types.KeyboardButton(text="4+ часов")
    button_back = telebot.types.KeyboardButton(text="Назад")
    keyboard.add(button_short, button_medium)
    keyboard.add(button_long, button_longest)
    keyboard.add(button_back)
    return keyboard

# Клавиатура для ввода интересов
def create_keypoints_input_keyboard():
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_skip = telebot.types.KeyboardButton(text="Пропустить")
    button_back = telebot.types.KeyboardButton(text="Назад")
    keyboard.add(button_skip)
    keyboard.add(button_back)
    return keyboard

# Получает строкой адрес из координат для проверки
def address_from_latitude_longitude(latitude, longitude):
    try:
        geolocator = Nominatim(user_agent="smart_city_guide_nn")
        coordinates = (latitude, longitude)
        location = geolocator.reverse(coordinates, timeout=10)
        return location.address if location else "Адрес не определен"
    except (GeocoderTimedOut, GeocoderServiceError):
        return "Ошибка определения адреса"
    except Exception:
        return "Адрес не определен"


def check_address(location):
    """
    Проверяет, находится ли адрес в Нижнем Новгороде или его окрестностях.
    """
    if not location:
        return False
    
    location_lower = location.lower()
    
    nn_keywords = [
        'нижний новгород', 
        'nizhny novgorod',
        'нижний',
        'новгород',
        'городской округ нижний новгород'
    ]
    
    # Проверяем наличие любого из ключевых слов
    return any(keyword in location_lower for keyword in nn_keywords)


# Геокодирование адреса из текста
def geocode_address_from_text(address_text):
    try:
        geolocator = Nominatim(user_agent="smart_city_guide_nn")
        
        # Добавляем город, если не указан
        if not any(word in address_text.lower() for word in ['нижний новгород',  'nizhny novgorod', 'нижний', 'новгород']):
            full_query = f"Нижний Новгород, {address_text}"
        else:
            full_query = address_text

        location = geolocator.geocode(full_query, timeout=10)
        
        if location:
            lat, lon = location.latitude, location.longitude
            # Проверяем, что в регионе Нижнего Новгорода
            if 55.8 <= lat <= 56.8 and 43.0 <= lon <= 44.8:
                return (lat, lon), location.address
        
        return None, None
    except Exception:
        return None, None

"""Состояния бота:
    - start_state - Начало работы и помощь
    - user_address_input_state - Ввод местоположения
    - estimated_time_input_state - Ввод времени прогулки
    - user_keypoint_state - Ввод интересов
    - generate_route_state - Генерация маршрута
"""

@bot.message_handler(commands=['start', 'help'])
def start_state(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    session.current_state = 0

    welcome_text = """
🏛️ Добро пожаловать в Кремлик - ваш персональный гид по Нижнему Новгороду!

Я помогу вам составить оптимальный маршрут для прогулки по интересным местам города с учетом:
• Ваших интересов
• Доступного времени
• Начального местоположения

Нажмите «Начало работы», чтобы начать!
    """

    keyboard = create_default_keyboard()
    bot.send_message(chat_id, welcome_text, reply_markup=keyboard)

@bot.message_handler(func=lambda message: message.text == 'Начало работы')
def user_address_input_state(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    session.current_state = 1

    # Загружаем модель при первом использовании
    if session.trained_model is None:
        try:
            bot.send_message(chat_id, "🔄 Загружаем данные о местах...")
            trainer = SmartCityGuideTrainer("cultural_objects_mnn.xlsx")
            session.trained_model = trainer.get_trained_model()
            bot.send_message(chat_id, "✅ Данные успешно загружены!")
        except Exception as e:
            bot.send_message(chat_id, f"❌ Ошибка загрузки данных: {e}")
            return

    address_text = """
📍 Укажите ваше начальное местоположение:

Вы можете:
• Отправить геотег (рекомендуется)
• Написать адрес текстом (например: «Большая Покровская улица, 12»)
• Ввести координаты (например: «56.326887 44.005986»)
    """

    keyboard = create_address_input_keyboard()
    bot.send_message(chat_id, address_text, reply_markup=keyboard)

@bot.message_handler(func=lambda message: message.text == 'Помощь')
def button_help(message):
    chat_id = message.chat.id
    help_text = """
📖 Помощь по использованию бота:

1. **Начало работы** - запуск процесса создания маршрута
2. **Указание местоположения** - отправьте геотег или введите адрес
3. **Выбор времени** - укажите продолжительность прогулки
4. **Интересы** - опишите, что вас интересует (музеи, парки, архитектура и т.д.)

Примеры интересов:
• «исторические места и музеи»
• «парки и природные достопримечательности» 
• «архитектура и храмы»
• «современное искусство и галереи»

Если не знаете, что ввести - просто нажмите «Пропустить»
    """
    bot.send_message(chat_id, help_text)

@bot.message_handler(func=lambda message: message.text == 'Назад')
def button_back(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    
    if session.current_state > 0:
        session.current_state -= 1
    
    if session.current_state == 0:
        start_state(message)
    elif session.current_state == 1:
        user_address_input_state(message)
    elif session.current_state == 2:
        estimated_time_input_state(message)

def estimated_time_input_state(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    session.current_state = 2

    time_text = """
⏱️ Сколько времени планируете на прогулку?

Учитывается:
• Время на посещение мест
• Пешие переходы между точками
• Рекомендуемое время осмотра

Выберите подходящий вариант:
    """

    keyboard = create_estimated_time_input_keyboard()
    bot.send_message(chat_id, time_text, reply_markup=keyboard)

def user_keypoint_state(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    session.current_state = 3

    interests_text = """
🎯 Что вас интересует?

Опишите ваши предпочтения через запятую или короткими фразами.

Примеры:
• музеи, искусство, история
• парки, природа, прогулки  
• архитектура, храмы, памятники
• кафе, рестораны, гастрономия
• развлечения, шопинг

Или нажмите «Пропустить» для маршрута по популярным местам
    """

    keyboard = create_keypoints_input_keyboard()
    bot.send_message(chat_id, interests_text, reply_markup=keyboard)
    bot.register_next_step_handler(message, handle_user_keypoint)

def handle_user_keypoint(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    
    if message.text == 'Пропустить':
        session.user_interests = "достопримечательности и интересные места"
    else:
        session.user_interests = message.text
    
    generate_route_state(message)

def generate_route_state(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    
    bot.send_message(chat_id, "🗺️ Составляем ваш персональный маршрут...")
    
    try:
        # Создаем планировщик с обученной моделью
        planner = SmartCityGuidePlanner(session.trained_model)
        
        # Используем методы планировщика для генерации маршрута
        trainer = SmartCityGuideTrainer.__new__(SmartCityGuideTrainer)
        trainer.model = session.trained_model['model']
        trainer.places = session.trained_model['places']
        trainer.place_embeddings = session.trained_model['place_embeddings']
        
        # Находим подходящие места
        relevant_places = trainer.find_optimal_places(
            session.user_interests, 
            session.min_time, 
            session.max_time, 
            session.start_coords
        )
        
        if not relevant_places:
            bot.send_message(chat_id, "❌ Не найдено подходящих мест для вашего запроса")
            start_state(message)
            return
        
        # Генерируем план маршрута (теперь возвращает список сообщений)
        messages = trainer.generate_route_plan(
            relevant_places, 
            session.min_time, 
            session.max_time, 
            session.user_interests, 
            session.start_location, 
            session.start_coords
        )
        
        # Отправляем каждое сообщение отдельно
        for msg in messages:
            # Если сообщение слишком длинное, разбиваем его
            if len(msg) > 4096:
                parts = [msg[i:i+4096] for i in range(0, len(msg), 4096)]
                for part in parts:
                    bot.send_message(chat_id, part)
            else:
                bot.send_message(chat_id, msg)
            
            # Небольшая задержка между сообщениями для лучшего восприятия
            import time
            time.sleep(0.5)
        
        
        keyboard = create_default_keyboard()
        bot.send_message(chat_id, "🔄 Хотите составить новый маршрут?", reply_markup=keyboard)
        session.current_state = 0
        
        
    except Exception as e:
        bot.send_message(chat_id, f"❌ Ошибка при генерации маршрута: {e}")
        start_state(message)

# Обработка текстового ввода адреса
@bot.message_handler(content_types=['text'], func=lambda message: get_user_session(message.chat.id).current_state == 1)
def handle_text_address(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    
    if message.text == 'Назад':
        button_back(message)
        return
        
    bot.send_message(chat_id, "🔍 Определяем координаты...")
    
    # Пробуем геокодировать адрес
    coords, address = geocode_address_from_text(message.text)
    
    if coords and address:
        session.start_coords = coords
        session.start_location = address
        bot.send_message(chat_id, f"✅ Местоположение определено: {address}")
        estimated_time_input_state(message)
    else:
        bot.send_message(chat_id, "❌ Не удалось определить адрес. Попробуйте еще раз или отправьте геотег.")
        user_address_input_state(message)

# Обработка выбора времени
@bot.message_handler(content_types=['text'], func=lambda message: get_user_session(message.chat.id).current_state == 2)
def handle_time_selection(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    
    if message.text == 'Назад':
        button_back(message)
        return
        
    time_mapping = {
        '1-2 часа': (1.0, 2.0),
        '2-3 часа': (2.0, 3.0),
        '3-4 часа': (3.0, 4.0),
        '4+ часов': (4.0, 6.0)
    }
    
    if message.text in time_mapping:
        session.min_time, session.max_time = time_mapping[message.text]
        bot.send_message(chat_id, f"⏱️ Время прогулки: {session.min_time}-{session.max_time} часов")
        user_keypoint_state(message)
    else:
        bot.send_message(chat_id, "❌ Пожалуйста, выберите время из предложенных вариантов")

# Обработка геотега
@bot.message_handler(content_types=['location'])
def handle_location(message):
    chat_id = message.chat.id
    session = get_user_session(chat_id)
    
    if session.current_state != 1:
        return
        
    latitude = message.location.latitude
    longitude = message.location.longitude
    
    address = address_from_latitude_longitude(latitude, longitude)
    
    if check_address(address):
        session.start_coords = (latitude, longitude)
        session.start_location = address
        bot.send_message(chat_id, f"✅ Местоположение определено: {address}")
        estimated_time_input_state(message)
    else:
        bot.send_message(chat_id, f"❌ Указанное местоположение находится вне Нижнего Новгорода: {address}")
        user_address_input_state(message)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"Использование памяти: {memory_mb:.2f} МБ")


# Запуск бота
if __name__ == "__main__":
    print("Бот запущен...")
    bot.infinity_polling()