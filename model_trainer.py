import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import re
from geopy.distance import geodesic
import networkx as nx

class SmartCityGuideTrainer:
    def __init__(self, excel_file_path):
        self.model = SentenceTransformer('cointegrated/rubert-tiny2')
        self.places = self.load_places_from_excel(excel_file_path)
        if self.places:
            self.place_embeddings = self.precompute_place_embeddings()
    
    def load_places_from_excel(self, file_path):
        """Загрузка мест из Excel файла"""
        try:
            df = pd.read_excel(file_path)
            places = []

            for index, row in df.iterrows():
                if pd.isna(row['id']) or pd.isna(row['title']):
                    continue

                coords = self.parse_coordinates(str(row['coordinate']))
                if not coords:
                    continue

                place = {
                    'id': int(row['id']),
                    'address': str(row['address']) if pd.notna(row['address']) else "Адрес не указан",
                    'coords': coords,
                    'description': str(row['description']) if pd.notna(row['description']) else str(row['title']),
                    'name': str(row['title']),
                    'url': str(row['url']) if 'url' in df.columns and pd.notna(row['url']) else ""
                }

                if 'category_id' in df.columns and pd.notna(row['category_id']):
                    place['category_id'] = int(row['category_id'])

                places.append(place)

            return places

        except Exception as e:
            print(f"Ошибка при загрузке Excel: {e}")
            return []

    def parse_coordinates(self, coord_str):
        """Парсинг координат из строки (POINT или lat,lon)"""
        try:
            if pd.isna(coord_str) or coord_str == 'nan':
                return None

            clean_str = str(coord_str).strip()

            if clean_str.startswith('POINT'):
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', clean_str)
                if len(numbers) >= 2:
                    lon = float(numbers[0])
                    lat = float(numbers[1])
                    if 43.0 <= lon <= 44.5 and 56.0 <= lat <= 56.5:
                        return (lat, lon)

            elif ',' in clean_str or ' ' in clean_str:
                parts = re.split(r'[,\s]+', clean_str)
                if len(parts) >= 2:
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        if 56.0 <= lat <= 56.5 and 43.0 <= lon <= 44.5:
                            return (lat, lon)
                    except ValueError:
                        pass

            return None

        except:
            return None

    def precompute_place_embeddings(self):
        """Предварительное вычисление эмбеддингов"""
        place_texts = [f"{place['name']} {place['description']}" for place in self.places]
        return self.model.encode(place_texts, convert_to_tensor=True, show_progress_bar=False)

    def calculate_walking_time(self, distance_km):
        walking_speed = 4.0  # км/ч
        walking_time_minutes = (distance_km / walking_speed) * 60
        return max(5, min(120, walking_time_minutes))

    def format_time_display(self, time_minutes):
        if time_minutes < 60:
            return f"{time_minutes:.0f} мин"
        else:
            return f"{time_minutes / 60:.1f} ч"

    def calculate_total_route_time(self, places, start_coords):
        if not places:
            return 0, 0, 0

        total_visit_time = sum(place['time_required'] for place in places)
        total_walking_time_min = 0
        current_coords = start_coords

        # От старта до первого места
        first_dist = geodesic(start_coords, places[0]['coords']).km
        total_walking_time_min += self.calculate_walking_time(first_dist)
        current_coords = places[0]['coords']

        # Между местами
        for i in range(1, len(places)):
            dist = geodesic(current_coords, places[i]['coords']).km
            total_walking_time_min += self.calculate_walking_time(dist)
            current_coords = places[i]['coords']

        total_walking_hours = total_walking_time_min / 60
        total_route_time = total_visit_time + total_walking_hours
        return total_route_time, total_walking_time_min, total_walking_hours

    def optimize_route_sequence(self, places, start_coords):
        if len(places) <= 1:
            return places

        all_points = [start_coords] + [place['coords'] for place in places]
        G = nx.Graph()

        for i, point in enumerate(all_points):
            G.add_node(i, pos=point)

        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                dist = geodesic(all_points[i], all_points[j]).km
                G.add_edge(i, j, weight=dist)

        def greedy_tsp(graph, start_node):
            unvisited = set(graph.nodes())
            unvisited.remove(start_node)
            tour = [start_node]
            current = start_node
            while unvisited:
                next_node = min(unvisited, key=lambda n: graph[current][n]['weight'])
                tour.append(next_node)
                unvisited.remove(next_node)
                current = next_node
            return tour

        tour = greedy_tsp(G, 0)
        return [places[i - 1] for i in tour[1:]]

    def find_optimal_places(self, user_interests, min_time, max_time, start_coords):
        if not self.places:
            return []

        query_embedding = self.model.encode(user_interests, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.place_embeddings)[0]

        places_with_scores = []
        for i, place in enumerate(self.places):
            dist_to_start = geodesic(start_coords, place['coords']).km
            walk_time_to_place = self.calculate_walking_time(dist_to_start) / 60  # время перехода в часах
            
            places_with_scores.append({
                **place,
                'similarity_score': cosine_scores[i].item(),
                'distance_to_start': dist_to_start,
                'walk_time_to_place': walk_time_to_place,
                'time_required': self.calculate_time_for_place(place, cosine_scores[i].item())
            })

        # Сортируем по релевантности и расстоянию
        places_with_scores.sort(key=lambda x: (-x['similarity_score'], x['distance_to_start']))

        best_combination = []
        best_time_diff = float('inf')

        # Пробуем разные количества мест от 3 до 5
        for target_count in [5, 4, 3]:  # Сначала пробуем 5 мест, потом 4, потом 3
            combination = []
            current_total_time = 0
            
            for place in places_with_scores:
                if len(combination) >= target_count:
                    break
                
                if place in combination:
                    continue
                    
                # Время для этого места (посещение + переход)
                place_total_time = place['time_required']
                
                if not combination:
                    # Первое место - добавляем переход от старта
                    place_total_time += place['walk_time_to_place']
                else:
                    # Для последующих мест - переход от предыдущего места
                    last_place = combination[-1]
                    dist_between = geodesic(last_place['coords'], place['coords']).km
                    walk_time_between = self.calculate_walking_time(dist_between) / 60
                    place_total_time += walk_time_between
                
                # Строгая проверка: общее время не должно превышать max_time
                if current_total_time + place_total_time <= max_time:
                    combination.append(place)
                    current_total_time += place_total_time

            # Проверяем, что набрали нужное количество мест и время подходит
            if len(combination) >= 3:  # Минимум 3 места
                total_route_time, _, _ = self.calculate_total_route_time(combination, start_coords)
                
                # Комбинация должна быть не меньше min_time и не больше max_time
                if min_time <= total_route_time <= max_time:
                    time_diff = abs(total_route_time - (min_time + max_time) / 2)
                    if time_diff < best_time_diff:
                        best_combination = combination
                        best_time_diff = time_diff

        # Если не нашли подходящую комбинацию, берем максимум мест которые влезают в время
        if not best_combination:
            best_combination = []
            current_time = 0
            
            for place in places_with_scores:
                if len(best_combination) >= 5:  # Максимум 5 мест
                    break
                    
                if place in best_combination:
                    continue
                    
                place_total_time = place['time_required']
                if not best_combination:
                    place_total_time += place['walk_time_to_place']
                else:
                    last_place = best_combination[-1]
                    dist_between = geodesic(last_place['coords'], place['coords']).km
                    walk_time_between = self.calculate_walking_time(dist_between) / 60
                    place_total_time += walk_time_between
                
                # Проверяем ограничение по времени
                if current_time + place_total_time <= max_time:
                    best_combination.append(place)
                    current_time += place_total_time

        # Гарантируем минимум 3 места (если есть хотя бы 3 места в базе)
        if len(best_combination) < 3 and len(places_with_scores) >= 3:
            # Добавляем недостающие места из топовых по релевантности
            for place in places_with_scores:
                if len(best_combination) >= 3:
                    break
                if place not in best_combination:
                    best_combination.append(place)

        return best_combination  # Возвращаем от 3 до 5 мест

    def calculate_time_for_place(self, place, similarity_score):
       
        if similarity_score > 0.7:
            return 0.4  
        elif similarity_score > 0.5:
            return 0.3  
        elif similarity_score > 0.3:
            return 0.25  
        else:
            return 0.2   

    def analyze_place_features(self, place):
        """Анализ особенностей места с более точной классификацией"""
        features = []
        description_lower = place['description'].lower()
        name_lower = place['name'].lower()
        full_text = name_lower + " " + description_lower

        # Более точные проверки с учетом контекста
        if any(word in full_text for word in ['исторический', 'историко-', 'историческое', 'древний', 'старинный', 'кремль', 'памятник истории']):
            features.append('исторический')
        elif any(word in full_text for word in ['истори', 'древн', 'старин', 'кремл']):
            features.append('исторический')

        if any(word in full_text for word in ['музей', 'галерея', 'выставка', 'культурный центр', 'центр культуры']):
            features.append('культурный')
        elif any(word in full_text for word in ['выставк', 'коллекц', 'культур', 'искусств']):
            features.append('культурный')

        if any(word in full_text for word in ['архитектурный памятник', 'памятник архитектуры', 'архитектурный ансамбль']):
            features.append('архитектурный')
        elif any(word in full_text for word in ['архитектур', 'здание', 'постройка', 'дворец', 'особняк', 'усадьба']):
            features.append('архитектурный')

        if any(word in full_text for word in ['природный парк', 'ботанический сад', 'дендрарий', 'заповедник']):
            features.append('природный')
        elif any(word in full_text for word in ['парк', 'сад', 'сквер', 'природ', 'озеро', 'река', 'набережная']):
            features.append('природный')

        if any(word in full_text for word in ['смотровая площадка', 'панорамный вид', 'обзорная площадка']):
            features.append('панорамный')
        elif any(word in full_text for word in ['вид', 'панорам', 'смотров', 'обзор']):
            features.append('панорамный')

        if any(word in full_text for word in ['храмовый комплекс', 'монастырский комплекс', 'святое место']):
            features.append('религиозный')
        elif any(word in full_text for word in ['церковь', 'собор', 'храм', 'монастырь', 'мечеть', 'синагога', 'часовня']):
            features.append('религиозный')

        if any(word in full_text for word in ['развлекательный комплекс', 'центр развлечений', 'аквапарк', 'парк аттракционов']):
            features.append('развлекательный')
        elif any(word in full_text for word in ['развлечен', 'аттракцион', 'кинотеатр', 'концертный']):
            features.append('развлекательный')

        if any(word in full_text for word in ['ресторан', 'кафе', 'кофейня', 'гастрономический', 'кулинарный']):
            features.append('гастрономический')

        if any(word in full_text for word in ['образовательный центр', 'научный центр', 'планетарий', 'обсерватория']):
            features.append('образовательный')
        elif any(word in full_text for word in ['университет', 'институт', 'библиотека', 'лекционный']):
            features.append('образовательный')

        if any(word in full_text for word in ['торговый центр', 'торгово-развлекательный', 'универмаг', 'торговая галерея']):
            features.append('торговый')
        elif any(word in full_text for word in ['магазин', 'бутик', 'рынок', 'аркада']):
            features.append('торговый')

        if any(word in full_text for word in ['спортивный комплекс', 'стадион', 'арена', 'физкультурно-оздоровительный']):
            features.append('спортивный')

        if any(word in full_text for word in ['детский центр', 'развивающий центр', 'игровая площадка', 'детский городок']):
            features.append('детский')

        if any(word in full_text for word in ['романтическое место', 'место для свиданий', 'аллея влюбленных']):
            features.append('романтический')

        if any(word in full_text for word in ['технический музей', 'индустриальный памятник', 'промышленный комплекс']):
            features.append('технический')

        if any(word in full_text for word in ['военный мемориал', 'музей военной техники', 'крепость', 'форт']):
            features.append('военный')

        if any(word in full_text for word in ['литературный музей', 'дом-музей писателя', 'литературное место']):
            features.append('литературный')

        if any(word in full_text for word in ['концертный зал', 'филармония', 'оперный театр', 'музыкальный театр']):
            features.append('музыкальный')

        if any(word in full_text for word in ['центр современного искусства', 'арт-пространство', 'галерея современного искусства']):
            features.append('современное искусство')

        if any(word in full_text for word in ['уникальный', 'единственный в городе', 'особенный', 'неповторимый']):
            features.append('уникальный')

        if any(word in full_text for word in ['семейный отдых', 'для всей семьи', 'семейный центр']):
            features.append('семейный')

        return list(set(features))

    def generate_reason(self, place, user_interests):
        """Генерация персонализированных причин выбора места"""
        score = place['similarity_score']
        features = self.analyze_place_features(place)
        user_interests_lower = user_interests.lower()

        # Сопоставление интересов пользователя с особенностями мест
        interest_keywords = {
            'истори': 'исторический',
            'культур': 'культурный',
            'архитектур': 'архитектурный',
            'природ': 'природный',
            'вид': 'панорамный',
            'панорам': 'панорамный',
            'религи': 'религиозный',
            'развлечен': 'развлекательный',
            'еда': 'гастрономический',
            'кухн': 'гастрономический',
            'ресторан': 'гастрономический',
            'обучен': 'образовательный',
            'образован': 'образовательный',
            'наук': 'образовательный',
            'шопинг': 'торговый',
            'покуп': 'торговый',
            'спорт': 'спортивный',
            'фитнес': 'спортивный',
            'детск': 'детский',
            'ребен': 'детский',
            'романт': 'романтический',
            'любов': 'романтический',
            'техн': 'технический',
            'индустри': 'технический',
            'воен': 'военный',
            'литератур': 'литературный',
            'книг': 'литературный',
            'музык': 'музыкальный',
            'современ': 'современное искусство',
            'арт': 'современное искусство',
            'уникальн': 'уникальный',
            'необычн': 'уникальный',
            'семей': 'семейный'
        }

        # Находим наиболее релевантную особенность для интересов пользователя
        best_matching_feature = None
        for interest_key, feature in interest_keywords.items():
            if interest_key in user_interests_lower and feature in features:
                best_matching_feature = feature
                break

        # Если нашли прямое соответствие, даем персонализированную причину
        if best_matching_feature:
            personalized_reasons = {
                'исторический': f"идеально соответствует вашему интересу к истории, {place['name']} представляет важные исторические объекты",
                'культурный': f"отвечает вашему запросу о культурных местах, здесь вы найдете интересные выставки и коллекции",
                'архитектурный': f"позволяет оценить архитектурные шедевры, которые вы ищете",
                'природный': f"предлагает прекрасные природные ландшафты для отдыха на свежем воздухе",
                'панорамный': f"открывает великолепные виды на город, что соответствует вашему желанию увидеть панорамы",
                'религиозный': f"знакомит с духовным наследием и религиозной архитектурой",
                'развлекательный': f"предлагает разнообразные развлечения для приятного времяпрепровождения",
                'гастрономический': f"позволяет насладиться местной кухней и гастрономическими особенностями",
                'образовательный': f"расширяет кругозор и предлагает познавательные программы",
                'торговый': f"идеально подходит для шопинга и покупок, которые вас интересуют",
                'спортивный': f"соответствует вашему активному образу жизни и спортивным интересам",
                'детский': f"создан специально для семей с детьми и предлагает развивающие программы",
                'романтический': f"создает особую атмосферу для романтической прогулки",
                'технический': f"демонстрирует технические достижения и инновации",
                'военный': f"рассказывает о военной истории и героическом прошлом",
                'литературный': f"связан с литературным наследием и творчеством известных писателей",
                'музыкальный': f"представляет богатую музыкальную культуру и традиции",
                'современное искусство': f"отражает актуальные тенденции в современном искусстве",
                'уникальный': f"является уникальной достопримечательностью, которую стоит увидеть",
                'семейный': f"предлагает программы и условия для комфортного семейного отдыха"
            }
            return personalized_reasons.get(best_matching_feature, f"Соответствует вашему интересу к {best_matching_feature} местам")

        # Если прямое соответствие не найдено, но есть особенности места
        if features:
            general_reasons = {
                'исторический': "знакомит с богатой историей и культурным наследием города",
                'культурный': "представляет значительный культурный интерес и образовательную ценность",
                'архитектурный': "является выдающимся образцом архитектуры и градостроительства",
                'природный': "предлагает возможность отдохнуть в природной среде",
                'панорамный': "открывает захватывающие виды на городские ландшафты",
                'религиозный': "отражает духовные традиции и религиозное искусство",
                'развлекательный': "обеспечивает интересный и разнообразный досуг",
                'гастрономический': "позволяет познакомиться с местными кулинарными традициями",
                'образовательный': "способствует получению новых знаний и расширению кругозора",
                'торговый': "предлагает широкий выбор товаров и услуг",
                'спортивный': "поддерживает активный образ жизни и физическое развитие",
                'детский': "создает благоприятную среду для развития и отдыха детей",
                'романтический': "обладает особой атмосферой, подходящей для романтических встреч",
                'технический': "демонстрирует технологический прогресс и инженерные решения",
                'военный': "хранит память о военных событиях и подвигах",
                'литературный': "связан с литературной историей и творчеством",
                'музыкальный': "представляет музыкальное искусство в различных формах",
                'современное искусство': "отражает современные художественные тенденции",
                'уникальный': "обладает уникальными характеристиками и особой ценностью",
                'семейный': "предназначен для комфортного отдыха всей семьей"
            }
            
            # Используем первую особенность из списка
            primary_feature = features[0]
            return general_reasons.get(primary_feature, f"Интересное место с {primary_feature} особенностями")

        # Резервные варианты на основе релевантности
        if score >= 0.8:
            return "Имеет высокую релевантность вашему запросу и рекомендуется системой"
        elif score >= 0.6:
            return "Хорошо соответствует вашим интересам и популярно среди туристов"
        elif score >= 0.4:
            return "Может быть интересно в рамках вашего маршрута"
        else:
            return "Дополняет маршрут интересными впечатлениями"

    def generate_route_plan(self, places, min_time, max_time, user_interests, start_location, start_coords):
        if not places or len(places) < 3:
            if self.places:
                places = self.places[:5]
            else:
                return ["❌ Недостаточно данных для построения маршрута"]

        if len(places) > 5:
            places = places[:5]

        optimized_places = self.optimize_route_sequence(places, start_coords)
        total_route_time, total_walking_time_min, _ = self.calculate_total_route_time(optimized_places, start_coords)
        total_visit_time = sum(p['time_required'] for p in optimized_places)

        from datetime import datetime, timedelta
        current_time = datetime.now()
        
        messages = []
        
        # Первое сообщение - заголовок маршрута
        header_message = f"🎯 ВАШ ТУРИСТИЧЕСКИЙ МАРШРУТ: '{user_interests}'\n"
        header_message += f"📅 Дата: {current_time.strftime('%d.%m.%Y')}\n"
        header_message += f"⏰ Начало: {current_time.strftime('%H:%M')}\n"
        header_message += f"📍 Стартовая точка: {start_location}\n"
        header_message += f"⏱️ Общее время маршрута: {self.format_time_display(total_route_time * 60)}\n"
        header_message += f"🎯 Запрошенное время: {min_time}-{max_time} ч\n"
        header_message += f"🏛️ Количество мест: {len(optimized_places)}\n"
        messages.append(header_message)

        current_coords = start_coords
        
        # Сообщение о переходе от старта до первого места
        if optimized_places:
            first_dist = geodesic(start_coords, optimized_places[0]['coords']).km
            first_walk = self.calculate_walking_time(first_dist)
            walk_end = current_time + timedelta(minutes=first_walk)
            transition_message = f"🚶‍♂️ ПЕРЕХОД ОТ СТАРТА ДО ПЕРВОГО МЕСТА:\n"
            transition_message += f"🕐 {current_time.strftime('%H:%M')} - {walk_end.strftime('%H:%M')}\n"
            transition_message += f"📏 Расстояние: {first_dist:.1f} км\n"
            transition_message += f"⏱️ Время: {self.format_time_display(first_walk)}"
            messages.append(transition_message)
            current_time = walk_end

        # Сообщения для каждого места
        for i, place in enumerate(optimized_places, 1):
            place_message = f"\n{i}. 🏛️ {place['name']}\n"
            place_message += f"⭐ Релевантность: {place['similarity_score']:.1%}\n"
            
            time_min = place['time_required'] * 60
            time_disp = self.format_time_display(time_min)
            end_time = current_time + timedelta(minutes=time_min)
            
            place_message += f"🕐 Время посещения: {current_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}\n"
            place_message += f"⏱️ Длительность: {time_disp}\n"
            place_message += f"📍 Адрес: {place['address']}\n"
            place_message += f"📖 Описание: {place['description']}\n"
            place_message += f"💡 Почему включено: {self.generate_reason(place, user_interests)}\n"
            
            if place.get('url'):
                place_message += f"🌐 Ссылка: {place['url']}\n"
                
            messages.append(place_message)
            current_time = end_time
            current_coords = place['coords']

            # Сообщение о переходе к следующему месту (кроме последнего)
            if i < len(optimized_places):
                next_dist = geodesic(current_coords, optimized_places[i]['coords']).km
                next_walk = self.calculate_walking_time(next_dist)
                walk_end = current_time + timedelta(minutes=next_walk)
                transition_message = f"\n🚶‍♂️ ПЕРЕХОД К СЛЕДУЮЩЕМУ МЕСТУ:\n"
                transition_message += f"🕐 {current_time.strftime('%H:%M')} - {walk_end.strftime('%H:%M')}\n"
                transition_message += f"📏 Расстояние: {next_dist:.1f} км\n"
                transition_message += f"⏱️ Время: {self.format_time_display(next_walk)}"
                messages.append(transition_message)
                current_time = walk_end

        # Финальное сообщение с итогами
        final_message = f"\n📊 ИТОГИ МАРШРУТА:\n"
        final_message += f"🏛️ Время на посещение мест: {self.format_time_display(total_visit_time * 60)}\n"
        final_message += f"🚶‍♂️ Время на пешие переходы: {self.format_time_display(total_walking_time_min)}\n"
        final_message += f"⏱️ Общее время маршрута: {self.format_time_display(total_route_time * 60)}\n"
        final_message += f"🎯 Завершение: {current_time.strftime('%H:%M')}\n"
        final_message += f"\n✨ Приятной прогулки! ✨"
        messages.append(final_message)

        return messages

    def get_trained_model(self):
        """Возвращает обученную модель для использования"""
        return {
            'model': self.model,
            'places': self.places,
            'place_embeddings': self.place_embeddings
        }

if __name__ == "__main__":
    # Обучение модели на данных из Excel
    excel_file = "cultural_objects_mnn.xlsx"
    trainer = SmartCityGuideTrainer(excel_file)
    print("Модель успешно обучена и готова к использованию")