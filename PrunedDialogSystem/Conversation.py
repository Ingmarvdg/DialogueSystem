# imports

import RulebasedEstimator
import pandas as pd
import numpy as np
import string
import csv
import json
from numpy import random
from Levenshtein import distance
import MLClassifier as mlc


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


class Conversation:
    SENTENCES = {
        "hello1": "Hello! Welcome to the Ambrosia restaurant system. What kind of restaurant are you looking for?",
        "hello2": "Hi there! You can search for restaurants by area, price range or food type. What would you like?",
        "empty1": "Sorry, no restaurants matching your criteria were found. Would you like to try something else?",
        "repeat1": "Sorry for not being clear, I said: ",
        "noise1": "I'm sorry, I didn't get that. Could you repeat?",
        "bye1": "Goodbye!",
        "bye2": "Bye, maybe next time",
        "restart1": "Ok! Let's try again!",
        "thankyou1": "You're welcome!",
        "ack1": "So, what can I help you with?",
        "request1": "Please, tell me what kind of restaurant you are looking for.",
        "inform1": "Here's the restaurant's information: .",
        "area1": "In which area would you like to go?",
        "pricerange1": "What kind of price range do you prefer?",
        "type1": "What type of food would you like?",
        "confirm1": "Yes,that is correct.",
        "deny1": "No.",
        "noresults1": "I couldnt find any results, lets try again, what is it you are looking for?",
        "softreset1": "Something else is also fine, what would you like?",
        "tryagain1": 'Sorry, lets try again, what is it you are looking for?',
        'ended1': 'The dialog has ended.',
        'reqall1': "Please tell me the type of food, the area and the price range you are interested in.",
        'reqarea1': "In which area do you want to eat? Center, north, east, south or west.",
        'reqtype1': 'What type of food are you looking for?',
        'reqprice1': 'What price range are you looking for?',
        'reqmore1': 'Anything else?',
        'confirmsent1': "So the food you are looking for has to be ",
        'reqone1': " ",
        'sorry2': "I'm afraid I cant do that",
        'maxresponses': "Sorry I couldnt help you within the time limit, goodbye!"
    }

    EMPTY_PREFERENCES = {"food": [], "pricerange": [], "area": []}

    INDIFFERENT_UTTERANCES = ["dont care", "anything", "doesnt matter", "i dont mind", "whatever", "any"]

    DATABASE_PATH = 'ontology/restaurantinfo.csv'

    ONTOLOGY_PATH = 'ontology/ontology_dstc2.json'

    def __init__(self, classifier, levenshtein_dist, direct_suggest):

        self.state = 'hello'
        self.response = 'nothing'
        self.user_preferences = self.EMPTY_PREFERENCES
        self.restaurantSet = []
        self.suggestion = {}
        self.infoGiven = False
        self.topic_at_stake = {}
        self.n_responses = 0
        self.restaurants = read_csv_database(self.DATABASE_PATH)
        self.ontology_data = read_json_ontology(self.ONTOLOGY_PATH)

        # configuration options
        self.classifier = classifier
        self.direct_suggest = direct_suggest
        self.levenshtein_dist = levenshtein_dist

    #    ONTOLOGY HANDLING HELPER FUNCTIONS      #

    def get_word_matches(self, word, field):
        possibilities = []
        for value in self.ontology_data['informable'][field]:
            ls_distance = distance(value, word)
            if ls_distance == 0:
                return value
            # For example not to ask unnecessary question such as 'Did you mean west instead of east?'
            elif ls_distance <= self.levenshtein_dist:
                possibilities.append(value)

        for value in possibilities:
            ask = input("Did you mean " + value + " instead of " + word + "? (yes/no)\n")
            if ask == 'yes':
                return value
            else:
                continue
        return None

    # update to just return preferences from sentence, and not update the previous preferences
    def get_preferences(self, sentence):
        preferences = self.EMPTY_PREFERENCES
        # Split sentence into a list of words
        words = [word.strip(string.punctuation) for word in sentence.split()]
        food_preferences = []
        pricerange_preferences = []
        area = []

        # Compare each word from the utterance content with the ontology data using the Levenshtein distance
        # and adding to the preference list each word which distance is equal or less than 1.
        for word in words:
            food_preferences.append(self.get_word_matches(word, 'food'))
            pricerange_preferences.append(self.get_word_matches(word, 'pricerange'))
            area.append(self.get_word_matches(word, 'area'))

        # Some restaurant and foods have in their names more than one word. So here we just check if the information
        # from the ontology exist in the utterance content.
        for value in self.ontology_data['informable']['food']:
            if value in sentence:
                food_preferences.append(value)
                break

        for indifferent_word in self.INDIFFERENT_UTTERANCES:
            if indifferent_word in sentence:
                if not preferences['food']:
                    preferences['food'] = ['any']
                if not preferences['pricerange']:
                    preferences['pricerange'] = ['any']
                if not preferences['area']:
                    preferences['area'] = ['any']

        # Make a distinct list
        food_preferences = list(dict.fromkeys(food_preferences))

        # Insert list to another
        preferences['food'][-1:-1] = list(filter(None, food_preferences))
        preferences['pricerange'][-1:-1] = list(filter(None, pricerange_preferences))
        # preferences['restaurantname'][-1:-1] = list(filter(None, name))
        preferences['area'][-1:-1] = list(filter(None, area))

        # Return the updated preferences
        return preferences

    def get_suggestion(self):
        # We search based on the area, price and food. We will return the
        # first restaurant which it will satisfy our query. In order not to get the same results over and over we would
        # also randomise the restaurants.
        prefs = self.user_preferences
        restaurants = self.restaurants

        if prefs['food'] and prefs['area'] and prefs['pricerange']:
            random.shuffle(self.restaurants)
            keys_with_value = []
            for key in prefs:
                if prefs[key] == ['any']:
                    continue
                elif prefs[key]:
                    keys_with_value.append(key)
            for restaurant in restaurants:
                count = 0
                for key in keys_with_value:
                    if restaurant[key] in prefs[key]:
                        count += 1
                if count == len(keys_with_value):
                    return restaurant

        # If we got here it means that the combination of all the the preferences didn't satisfy our query.
        # We will suggest something based on the pricerange and the food.
        if prefs['food'] and prefs['pricerange']:
            random.shuffle(restaurants)
            keys_with_value = []
            for key in prefs:
                if key == 'restaurantname' or key == 'area' or prefs[key] == ['any']:
                    continue
                if prefs[key]:
                    keys_with_value.append(key)
            for restaurant in restaurants:
                count = 0
                for key in keys_with_value:
                    if restaurant[key] in prefs[key]:
                        count += 1
                if count == len(keys_with_value):
                    return restaurant

        # We will suggest something based on the food and the area.
        if prefs['food'] and prefs['area']:
            random.shuffle(restaurants)
            keys_with_value = []
            for key in prefs:
                if key == 'restaurantname' or key == 'pricerange' or prefs[key] == ['any']:
                    continue
                if prefs[key]:
                    keys_with_value.append(key)
            for restaurant in restaurants:
                count = 0
                for key in keys_with_value:
                    if restaurant[key] in prefs[key]:
                        count += 1
                if count == len(keys_with_value):
                    return restaurant

        # Now we should suggest based on food only then on pricerange and last on the area.
        random.shuffle(restaurants)
        if prefs['food']:
            for preference in prefs['food']:
                restaurant = next((restaurant for restaurant in restaurants if restaurant['food'] == preference), None)
                if restaurant is not None:
                    return restaurant

        random.shuffle(restaurants)
        if prefs['pricerange']:
            for preference in prefs['pricerange']:
                restaurant = next((restaurant for restaurant in restaurants if restaurant['pricerange'] == preference),
                                  None)
                if restaurant is not None:
                    return restaurant
                if restaurant is None and prefs['pricerange'] == ['any']:
                    return restaurants[0]

        random.shuffle(restaurants)
        if prefs['area']:
            for preference in prefs['area']:
                restaurant = next((restaurant for restaurant in restaurants if restaurant['area'] == preference), None)
                if restaurant is not None:
                    return restaurant
                if restaurant is None and prefs['area'] == ['any']:
                    return restaurants[0]
        return

    def check_suggestion(self, suggestions, utterance_content):
        keywords = ['food', 'area', 'price']
        for key in keywords:
            if key in utterance_content:
                for suggestion in suggestions[key]:
                    if suggestion in utterance_content:
                        return True
        return False

    @staticmethod
    def get_dialog_act(sentence, classifier):
        if classifier == 'rule':
            dialog_act = RulebasedEstimator.predict(sentence)
        elif isinstance(classifier, pd.core.series):
            dialog_act = np.random.choice(list(classifier.index.values), 1, list(classifier.values))
        else:
            dialog_act = mlc.label_single(sentence)
        return dialog_act

    # method that will update the user preferences with the topic that's at stake
    def update_preferences(self):
        for topic in self.topic_at_stake:
            self.user_preferences[topic] = list(set(self.user_preferences[topic] + self.topic_at_stake[topic]))
        return

    #   SENTENCE GENERATION HELPER FUNCTIONS    #

    def get_request_sent(self):
        prefs = self.user_preferences

        if not prefs['area'] and not prefs['pricerange'] and not prefs['food']:
            request = self.SENTENCES['reqall1']
        elif not prefs['area']:
            request = self.SENTENCES['reqarea1']
        elif not prefs['food']:
            request = self.SENTENCES['reqtype1']
        elif not prefs['pricerange']:
            request = self.SENTENCES['reqprice1']
        else:
            request = self.SENTENCES['reqmore1']

        return request

    def get_confirm_sent(self):
        stake = self.topic_at_stake
        varying_sents = []

        if stake['food']:
            if stake['food'] == ['any']:
                varying_sents.append('any food')
            else:
                varying_sents.append(' or '.join(stake['food']))
        if stake['pricerange']:
            if stake['pricerange'] == ['any']:
                varying_sents.append('any price range')
            else:
                varying_sents.append(' or '.join(stake['pricerange']) + " price")
        if stake['area']:
            if stake['area'] == ['any']:
                varying_sents.append('in any area')
            else:
                varying_sents.append("in the " + ' or '.join(stake['area']))

        confirm_sent = self.SENTENCES['confirmsent1'] + ' and '.join(varying_sents) + "?"

        return confirm_sent

    #   STATE FUNCTIONS     #

    def state_hello(self, sentence, act):
        response = ''
        new_state = 'hello'

        if act == 'ack' or act == 'affirm':
            new_state = 'request'
            response = self.SENTENCES['ack1']

        if act == 'confirm':
            new_state = 'request'
            response = self.SENTENCES['request1']

        if act == 'negate':
            new_state = 'bye'
            response = self.SENTENCES['bye2']

        if act == 'reqmore' or act == 'request':
            response = self.SENTENCES['noise1']

        if act == 'inform' or act == 'deny' or act == 'reqalts':
            self.topic_at_stake = self.get_preferences(sentence)
            new_state = 'confirm'
            response = self.get_confirm_sent()

        return response, new_state

    def state_request(self, sentence, act):
        response = ''
        new_state = 'request'

        if act == 'ack' or act == 'affirm':
            response = self.SENTENCES['ack1']

        if act == 'confirm' or act == 'reqalts' or act == 'inform' or act == 'request':
            self.topic_at_stake = self.get_preferences(sentence)
            if len(self.topic_at_stake) < 3:
                response = self.SENTENCES['reqall1']

            elif len(self.topic_at_stake) > 1:
                response = self.SENTENCES['reqone1']

            else:
                response = self.get_confirm_sent()
                new_state = 'confirm'

        if act == 'deny' or act == 'negate':
            response = self.SENTENCES['bye1']
            new_state = 'bye'

        if act == 'reqmore':
            response = self.SENTENCES['noise1']

        return response, new_state

    def state_confirm(self, sentence, act):
        response = ''
        new_state = 'confirm'

        if act == 'ack' or act == 'affirm':
            self.update_preferences()

            if self.direct_suggest:
                self.suggestion = self.get_suggestion()

                name = self.suggestion['restaurantname']
                price = self.suggestion['pricerange']
                area = self.suggestion['area']

                response = f"What about {name}? Its {price} and is in the {area}."
                new_state = 'inform'

            if not self.direct_suggest:
                if not self.user_preferences['area'] \
                        or not self.user_preferences['pricerange'] \
                        or not self.user_preferences['food']:

                    response = self.get_request_sent()
                    new_state = 'request'

                else:
                    self.suggestion = self.get_suggestion()

                    name = self.suggestion['restaurantname']
                    price = self.suggestion['pricerange']
                    area = self.suggestion['area']

                    response = f"What about {name}? Its {price} and is in the {area}."
                    new_state = 'inform'

        if act == 'confirm' or act == 'inform':
            response = self.get_confirm_sent()
            new_state = 'confirm'

        if act == 'reqalts':
            response = self.SENTENCES['softreset1']
            new_state = 'request'

        if act == 'deny' or act == 'negate':
            self.topic_at_stake = {}
            new_state = 'request'
            response = self.SENTENCES['tryagain1']

        return response, new_state

    def state_inform(self, sentence, act):
        response = ''
        new_state = 'inform'

        if act == 'inform':
            response = self.get_confirm_sent(sentence)
            new_state = 'confirm'

        if act == 'confirm':
            if self.check_suggestion(self.suggestion, sentence):
                response = self.SENTENCES['confirm1']
            else:
                new_state = 'confirm'
                response = self.get_confirm_sent(sentence)

        if act == 'deny' or act == 'negate' or act == 'reqmore':
            self.suggestion = self.get_suggestion()

            name = self.suggestion['restaurantname']
            price = self.suggestion['pricerange']
            area = self.suggestion['area']

            response = f"What about {name}, its {price} and is in the {area}."

        if act == 'ack' or act == 'affirm':
            if not self.infoGiven:
                response = f"The restaurants  phone number is {self.suggestion['phone']}," \
                           f" the address is {self.suggestion['addr']}, {self.suggestion['postcode']}."
                self.infoGiven = True
            else:
                response = 'goodbye'
                new_state = ''

        if act == 'request':
            response = self.get_inform_sent(sentence)

        if act == 'inform':
            new_state = 'confirm'

        return response, new_state

    def state_bye(self, sentence):
        response = self.SENTENCES['bye2']
        new_state = 'bye'

        return response, new_state

    def get_next_sentence(self, sentence=None):
        sentence.lower()

        switcher = {
            "hello": self.state_hello,
            "request": self.state_request,
            "confirm": self.state_confirm,
            "inform": self.state_inform,
            "bye": self.state_bye
        }

        # get dialog act
        act = self.get_dialog_act(sentence, self.classifier)

        # general responses
        if act == 'hello':
            self.state = 'request'
            self.response = self.SENTENCES['hello2']

        elif act == 'repeat':
            return self.SENTENCES['repeat1'] + self.response

        elif act == 'null':
            self.response = self.SENTENCES['noise1']
            return self.response

        elif act == 'bye':
            print(self.n_responses)
            return self.SENTENCES['bye1']

        elif act == 'restart':
            self.__init__()
            return self.SENTENCES['restart1']

        elif act == 'thankyou':
            return self.SENTENCES['thankyou1']

        else:
            func = switcher.get(self.state, lambda: "Invalid State")
            response, new_state = func(sentence, act)
            self.state = new_state
            self.response = response

        self.n_responses = self.n_responses + 1

        return self.response

    def start_conversation(self):

        print(self.SENTENCES['hello1'])
        sentence = input()
        while self.state != 'bye':
            print(self.get_next_sentence(sentence))
            sentence = input()

        return self.SENTENCES['ended1']
