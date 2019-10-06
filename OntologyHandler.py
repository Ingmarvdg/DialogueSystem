import csv
import json
import string
# If you encounter the following error "Microsoft Visual C++ 14.0 is required"
# you need to download Visual Studio toolbox
# https://stackoverflow.com/questions/29846087/microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat
from Levenshtein import distance
from Lib import random


def get_update_preference_data(word, ontology_data, type):
    for value in ontology_data['informable'][type]:
        if distance(value, word) == 0:
            return value
        # For example not to ask unnecessary question such as 'Did you mean west instead of east?'
        elif distance(value, word) <= 1 and \
                word not in ontology_data['informable']['food'] and \
                word not in ontology_data['informable']['pricerange'] and \
                word not in ontology_data['informable']['name'] and \
                word not in ontology_data['informable']['area']:
            ask = input("Did you mean " + value + " instead of " + word + "? (yes/no)\n")
            if ask == 'yes':
                return value
            else:
                continue


def read_csv_database(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        restaurants = list(reader)
        restaurants_header = restaurants.pop(0)
        dict_restaurant = []
        for restaurant in restaurants:
            i = 0
            temp = {}
            for header_name in restaurants_header:
                temp[header_name] = restaurant[i]
                i += 1
            dict_restaurant.append(temp)
        return dict_restaurant


def read_json_ontology(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
        return data


def extracting_preferences(utterance_content, ontology_data, preferences):
    # Split sentence into a list of words
    words = [word.strip(string.punctuation) for word in utterance_content.split()]
    print(words)
    food_preferences = []
    pricerange_preferences = []
    name = []
    area = []
    # Compare each word from the utterance content with the ontology data using the Levenshtein distance
    # and adding to the preference list each word which distance is equal or less than 1.
    for word in words:
        food_preferences.append(get_update_preference_data(word, ontology_data, 'food'))
        pricerange_preferences.append(get_update_preference_data(word, ontology_data, 'pricerange'))
        name.append(get_update_preference_data(word, ontology_data, 'name'))
        area.append(get_update_preference_data(word, ontology_data, 'area'))

    # Some restaurant and foods have in their names more than one word. So here we just check if the information
    # from the ontology exist in the utterance content.
    for value in ontology_data['informable']['food']:
        if value in utterance_content:
            food_preferences.append(value)
            break

    for value in ontology_data['informable']['name']:
        if value in utterance_content:
            name.append(value)
            break

    # Make a distinct list
    name = list(dict.fromkeys(name))
    food_preferences = list(dict.fromkeys(food_preferences))

    # Insert list to another
    preferences['food'][-1:-1] = list(filter(None, food_preferences))
    preferences['pricerange'][-1:-1] = list(filter(None, pricerange_preferences))
    preferences['restaurantname'][-1:-1] = list(filter(None, name))
    preferences['area'][-1:-1] = list(filter(None, area))

    # Return the updated preferences
    return preferences


def get_info_from_restaurant(preferences, restaurants):
    # If we know the restaurant name we just print their details as it is our primary goal.
    if preferences['restaurantname']:
        for preference_restaurant in preferences['restaurantname']:
            restaurant = next((restaurant for restaurant in restaurants if restaurant['restaurantname'] == preference_restaurant), None)
            return restaurant

    # If we don't have the restaurantname info then we search based on the area, price and food. We will return the
    # first restaurant which it will satisfy our query. In order not to get the same results over and over we would
    # also randomise the restaurants.
    random.shuffle(restaurants)
    keys_with_value = []
    for key in preferences:
        if key == 'restaurantname':
            continue
        if preferences[key]:
            keys_with_value.append(key)
    for restaurant in restaurants:
        count = 0
        for key in keys_with_value:
            if restaurant[key] in preferences[key]:
                count += 1
        if count == len(keys_with_value):
            return restaurant

    # If we got here it means that the combination of all the the preferences didn't satisfy our query.
    # We will suggest something based on the pricerange and the food.
    random.shuffle(restaurants)
    keys_with_value = []
    for key in preferences:
        if key == 'restaurantname' or key == 'area':
            continue
        if preferences[key]:
            keys_with_value.append(key)
    for restaurant in restaurants:
        count = 0
        for key in keys_with_value:
            if restaurant[key] in preferences[key]:
                count += 1
        if count == len(keys_with_value):
            return restaurant

    # Now we should suggest based on food only then on pricerange and last on the area.
    random.shuffle(restaurants)
    if preferences['food']:
        for preference in preferences['food']:
            restaurant = next((restaurant for restaurant in restaurants if restaurant['food'] == preference), None)
            return restaurant
    if preferences['pricerange']:
        for preference in preferences['pricerange']:
            restaurant = next((restaurant for restaurant in restaurants if restaurant['pricerange'] == preference), None)
            return restaurant
    if preferences['area']:
        for preference in preferences['area']:
            restaurant = next((restaurant for restaurant in restaurants if restaurant['area'] == preference), None)
            return restaurant

    return None




