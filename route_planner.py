from datetime import datetime, timedelta
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from model_trainer import SmartCityGuideTrainer

class SmartCityGuidePlanner:
    def __init__(self, trained_model):
        self.model = trained_model['model']
        self.places = trained_model['places']
        self.place_embeddings = trained_model['place_embeddings']
    
    def get_user_input(self):
        """Получение данных от пользователя"""
        print("\n1. Какие места вас интересуют?")
        user_interests = input("   Ваши интересы: ").strip()

        if not user_interests:
            user_interests = "достопримечательности и интересные места"

        print("\n2. Укажите время на прогулку (включая переходы между местами)")
        print("   Минимальное время (часов):")
        try:
            min_time = float(input("   От: ").strip())
        except:
            min_time = 2.0

        print("   Максимальное время (часов):")
        try:
            max_time = float(input("   До: ").strip())
        except:
            max_time = min_time + 2.0

        print("\n3. Укажите координаты начала маршрута (обязательно)")
        print("   Вы можете ввести:")
        print("     • Координаты: 56.326887 44.005986")
        print("     • Адрес: Большая Покровская улица, 12")

        while True:
            start_input = input("   Ваш ввод: ").strip()
            coords, location_name = self.parse_user_coordinates(start_input)

            if coords:
                return user_interests, min_time, max_time, coords, location_name
            else:
                print("   ❌ Не удалось определить начальную точку. Попробуйте снова.")

    def geocode_address(self, address_query):
        """
        Преобразует адрес в координаты с привязкой к Нижнему Новгороду.
        """
        try:
            query = address_query.strip()
            if not query:
                return None, None

            # Добавляем город, если не указан
            if not any(word in query.lower() for word in ['нижний новгород', 'nizhny novgorod', 'нижний', 'новгород']):
                full_query = f"Нижний Новгород, {query}"
            else:
                full_query = query

            geolocator = Nominatim(user_agent="smart_city_guide_nn")
            location = geolocator.geocode(full_query, timeout=10)

            if location:
                lat, lon = location.latitude, location.longitude
                # Расширенная зона Нижнего Новгорода
                if 55.8 <= lat <= 56.8 and 43.0 <= lon <= 44.8:
                    return (lat, lon), location.address
                else:
                    print(f"   Адрес найден, но вне региона Нижнего Новгорода.")
                    return None, None
            else:
                print("   Адрес не найден. Проверьте написание.")
                return None, None

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"   Ошибка геокодирования: сервер не отвечает.")
            return None, None
        except Exception as e:
            print(f"   Ошибка при геокодировании: {e}")
            return None, None

    def parse_user_coordinates(self, coord_str):
        """
        Поддерживает ввод координат ИЛИ адреса.
        """
        if not coord_str:
            return None, None

        # Попытка распарсить как координаты
        clean_str = re.sub(r'[,\s]+', ' ', coord_str).strip()
        parts = clean_str.split()

        if len(parts) == 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                if 55.8 <= lat <= 56.8 and 43.0 <= lon <= 44.8:
                    return (lat, lon), f"Координаты ({lat:.6f}, {lon:.6f})"
            except ValueError:
                pass  # Не координаты — пробуем как адрес

        # Иначе — геокодируем как адрес
        return self.geocode_address(coord_str)

    def calculate_walking_time(self, distance_km):
        walking_speed = 4.0  # км/ч
        walking_time_minutes = (distance_km / walking_speed) * 60
        return max(5, min(120, walking_time_minutes))

    def format_time_display(self, time_minutes):
        if time_minutes < 60:
            return f"{time_minutes:.0f} мин"
        else:
            return f"{time_minutes / 60:.1f} ч"

    def create_walk_plan(self):
        if not self.places:
            print("❌ Нет данных о местах")
            return

        user_interests, min_time, max_time, start_coords, start_location = self.get_user_input()
        print(f"\n Начальная точка: {start_location}")
        print(f" Координаты: {start_coords[0]:.6f}, {start_coords[1]:.6f}")

        # Используем методы из обученной модели для поиска мест
        trainer = SmartCityGuideTrainer.__new__(SmartCityGuideTrainer)
        trainer.model = self.model
        trainer.places = self.places
        trainer.place_embeddings = self.place_embeddings
        
        relevant_places = trainer.find_optimal_places(user_interests, min_time, max_time, start_coords)
        if not relevant_places:
            print("❌ Не найдено подходящих мест")
            return

        plan = trainer.generate_route_plan(relevant_places, min_time, max_time, user_interests, start_location, start_coords)
        print(plan)


if __name__ == "__main__":
    # Загрузка обученной модели и создание планера
    excel_file = "cultural_objects_mnn.xlsx"
    try:
        # Сначала обучаем модель
        trainer = SmartCityGuideTrainer(excel_file)
        trained_model = trainer.get_trained_model()
        
        # Затем используем планер
        guide = SmartCityGuidePlanner(trained_model)
        guide.create_walk_plan()
    except Exception as e:
        print(f"❌ Ошибка: {e}")