# imports

import RulebasedEstimator
import pandas as pd
import numpy as np
import string
# from Levenshtein import distance
from numpy import random
import Levenshtein

# conversation class
from OntologyHandler import read_csv_database, read_json_ontology


# levenstein placeholder
def distance(a, b):
    return 0


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
        'confirmtype1': "So the food you are looking for has to be ",
        'confirmprice1': "The price has to be ",
        'confirmarea1': "You want the restaurant to be in the ",
        'reqone1': " ",
        'sorry2': "I'm afraid I cant do that",
        'maxresponses': "Sorry I couldnt help you within the time limit, goodbye!"
    }

    EMPTY_PREFERENCES = {"food": [], "pricerange": [], "area": [], "name": []}

    RESTAURANT_DATA = []

    def __init__(self, classifier='rule',
                 database_path='ontology/restaurantinfo.csv',
                 ontology_path='ontology/ontology_dstc2.json',
                 confirmation_all=True,
                 info_per_utt="",
                 levenshtein_dist=5,
                 allow_restarts=True,
                 max_responses=np.inf,
                 response_uppercase=True,
                 utt_lowercase=True):

        self.state = 'hello'
        self.response = 'nothing'
        self.user_preferences = self.EMPTY_PREFERENCES
        self.restaurantSet = []
        self.suggestion = {}
        self.infoGiven = False
        self.topic_at_stake = {}
        self.n_responses = 0
        self.restaurants = read_csv_database(database_path)
        self.ontology_data = read_json_ontology(ontology_path)

        # configuration options
        self.classifier = classifier
        self.confirmation_all = confirmation_all
        self.info_per_utt = info_per_utt  # all: request all at once, empty: any number, one: request one at a time
        self.levenshtein_dist = levenshtein_dist
        self.allow_restarts = allow_restarts
        self.max_responses = max_responses
        self.response_uppercase = response_uppercase
        self.utt_lowercase = utt_lowercase

    #    ONTOLOGY HANDLING HELPER FUNCTIONS      #

    def get_word_matches(self, word, field):
        for value in self.ontology_data['informable'][field]:
            if distance(value, word) <= self.levenshtein_dist:
                return value
            # For example not to ask unnecessary question such as 'Did you mean west instead of east?'
            elif distance(value, word) > self.levenshtein_dist and \
                    word not in self.ontology_data['informable']['food'] and \
                    word not in self.ontology_data['informable']['pricerange'] and \
                    word not in self.ontology_data['informable']['name'] and \
                    word not in self.ontology_data['informable']['area']:
                ask = input("Did you mean " + value + " instead of " + word + "? (yes/no)\n")
                if ask == 'yes':
                    return value
                else:
                    continue

    # update to just return preferences from sentence, and not update the previous preferences
    def get_preferences(self, sentence):
        preferences = self.user_preferences
        # Split sentence into a list of words
        words = [word.strip(string.punctuation) for word in sentence.split()]
        food_preferences = []
        pricerange_preferences = []
        name = []
        area = []
        # Compare each word from the utterance content with the ontology data using the Levenshtein distance
        # and adding to the preference list each word which distance is equal or less than 1.
        for word in words:
            food_preferences.append(self.get_word_matches(word, 'food'))
            pricerange_preferences.append(self.get_word_matches(word, 'pricerange'))
            name.append(self.get_word_matches(word, 'name'))
            area.append(self.get_word_matches(word, 'area'))

        # Some restaurant and foods have in their names more than one word. So here we just check if the information
        # from the ontology exist in the utterance content.
        for value in self.ontology_data['informable']['food']:
            if value in sentence:
                food_preferences.append(value)
                break

        for value in self.ontology_data['informable']['name']:
            if value in sentence:
                name.append(value)
                break

        # Make a distinct list
        name = list(dict.fromkeys(name))
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
                if key == 'restaurantname':
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

        # If we got here it means that the combination of all the the preferences didn't satisfy our query.
        # We will suggest something based on the pricerange and the food.
        if prefs['food'] and prefs['pricerange']:
            random.shuffle(restaurants)
            keys_with_value = []
            for key in prefs:
                if key == 'restaurantname' or key == 'area':
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
                if key == 'restaurantname' or key == 'pricerange':
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
        if prefs['pricerange']:
            for preference in prefs['pricerange']:
                restaurant = next((restaurant for restaurant in restaurants if restaurant['pricerange'] == preference),
                                  None)
                if restaurant is not None:
                    return restaurant
        if prefs['area']:
            for preference in prefs['area']:
                restaurant = next((restaurant for restaurant in restaurants if restaurant['area'] == preference), None)
                if restaurant is not None:
                    return restaurant
        return None

    def check_suggestion(self, suggestions, utterance_content):
        keywords = ['food', 'area', 'price']
        for key in keywords:
            if key in utterance_content:
                for suggestion in suggestions[key]:
                    if suggestion in utterance_content:
                        return True
        return False

    def get_restaurant_information(self, restaurant):

        return restaurant

    @staticmethod
    def get_dialog_act(sentence, classifier):
        if classifier == 'rule':
            dialog_act = RulebasedEstimator.predict(sentence)
        elif isinstance(classifier, pd.core.series):
            dialog_act = np.random.choice(list(classifier.index.values), 1, list(classifier.values))
        else:
            dialog_act = classifier.predict(sentence)
        return dialog_act

    # method that will update the user preferences with the topic that's at stake
    def update_preferences(self):
        for topic in self.topic_at_stake:
            self.user_preferences[topic].append(self.topic_at_stake[topic])
        return

    #   SENTENCE GENERATION HELPER FUNCTIONS    #

    def get_request_sent(self):
        prefs = self.user_preferences

        if not prefs['area'] and not prefs['price_range'] and not prefs['type']:
            request = self.SENTENCES['reqall1']
        elif not prefs['area']:
            request = self.SENTENCES['reqarea1']
        elif not prefs['type']:
            request = self.SENTENCES['reqtype1']
        elif not prefs['price_range']:
            request = self.SENTENCES['reqprice1']
        else:
            request = self.SENTENCES['reqmore1']

        return request

    def get_confirm_sent(self):
        stake = self.topic_at_stake
        confirm = '_'

        if 'type' in stake:
            confirm = self.SENTENCES['confirmtype1'] + ''.join(stake['type'] + "?")
        if 'price_range' in stake:
            confirm = self.SENTENCES['confirmprice1'], stake['price_range']
        if 'area' in stake:
            confirm = self.SENTENCES['confirmarea1'] + ' '.join(stake['area']) + "?"

        return confirm

    def get_inform_sent(self, sentence):
        inform = 'information'
        return inform

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
            if self.info_per_utt == "all" and len(self.topic_at_stake) < 3:
                response = self.SENTENCES['reqall1']

            elif self.info_per_utt == "one" and len(self.topic_at_stake) > 1:
                response = self.SENTENCES['reqone1']

            elif self.confirmation_all:
                response = self.get_confirm_sent()
                new_state = 'confirm'
            else:
                self.update_preferences()

                if not self.user_preferences['area'] \
                        or not self.user_preferences['pricerange'] \
                        or not self.user_preferences['food']:

                    response = self.get_request_sent()
                    new_state = 'request'

                else:
                    self.suggestion = self.get_suggestion()

                    name = self.suggestion['name']
                    price = self.suggestion['price']
                    area = self.suggestion['area']

                    response = f"What about {name}, its {price} and is in the {area}."
                    new_state = 'inform'

        if act == 'deny' or 'negate':
            response = self.SENTENCES['bye1']
            new_state = 'bye'

        if act == 'reqmore':
            response = self.SENTENCES['noise1']

        return response, new_state

    def state_confirm(self, act):
        response = ''
        new_state = 'confirm'

        if act == 'ack' or act == 'affirm':
            self.update_preferences()

            if not self.user_preferences['area'] \
                    or not self.user_preferences['pricerange'] \
                    or not self.user_preferences['food']:

                response = self.get_request_sent()
                new_state = 'request'

            else:
                self.suggestion = self.get_suggestion()

                name = self.suggestion['name']
                price = self.suggestion['price']
                area = self.suggestion['area']

                response = f"What about {name}, its {price} and is in the {area}."
                new_state = 'inform'

        if act == 'confirm' or act == 'inform':
            response = self.get_confirm_sent()
            new_state = 'confirm'

        if act == 'reqalts':
            response = self.SENTENCES['softreset1']
            new_state = 'request'

        if act == 'deny' or 'negate':
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

        if act == 'deny' or 'negate' or 'reqmore':
            self.suggestion = self.get_suggestion()

            name = self.suggestion['name']
            price = self.suggestion['price']
            area = self.suggestion['area']

            response = f"What about {name}, its {price} and is in the {area}."

        if act == 'ack' or 'affirm':
            if not self.infoGiven:
                response = self.get_inform_sent(sentence)

            if self.infoGiven:
                response = 'goodbye'
                new_state = ''

        if act == 'request':
            response = self.get_inform_sent(sentence)

        if act == 'inform':
            new_state = 'confirm'

        return response, new_state

    def state_bye(self, sentence):
        response = self.SENTENCES['bye1']
        new_state = 'bye'

        return response, new_state

    def get_next_sentence(self, sentence=None):
        if self.utt_lowercase:
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

        print("current state is: " + self.state, "current act is: " + act)

        # check utt limit
        if self.n_responses > self.max_responses:
            self.state = "bye"
            self.response = self.SENTENCES['maxresponses']

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
            return self.SENTENCES['bye1']

        elif act == 'restart':
            if self.allow_restarts:
                self.__init__()
                return self.SENTENCES['restart1']
            else:
                return self.SENTENCES['sorry2']

        elif act == 'thankyou':
            return self.SENTENCES['thankyou1']

        else:
            func = switcher.get(self.state, lambda: "Invalid State")
            response, new_state = func(sentence, act)
            self.state = new_state
            self.response = response

        if self.response_uppercase:
            self.response = self.response.upper()

        print("new state is: " + self.state)

        return self.response

    def start_conversation(self):

        print(self.SENTENCES['hello1'])
        sentence = input()
        while self.state != 'bye':
            print(self.get_next_sentence(sentence))
            sentence = input()

        return self.SENTENCES['ended1']


#convo = Conversation(classifier='rule')
#convo.start_conversation()
