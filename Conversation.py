# imports

import RulebasedEstimator
import pandas as pd
import numpy as np


# conversation class
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
        'confirmprice1': " ",
        'confirmarea1': " ",
        'reqone1': " ",
        'sorry2': "I'm afraid I cant do that",
        'maxresponses': "Sorry I couldnt help you within the time limit, goodbye!"
    }

    EMPTY_PREFERENCES = {"food": [], "price_range": [], "area": []}

    RESTAURANT_DATA = []

    def __init__(self, classifier='rule',
                 confirmation_all=True,
                 info_per_utt="",
                 levenshtein_dist=1,
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

        # configuration options
        self.classifier = classifier
        self.confirmation_all = confirmation_all
        self.info_per_utt = info_per_utt  # all: request all at once, empty: any number, one: request one at a time
        self.levenshtein_dist = levenshtein_dist
        self.allow_restarts = allow_restarts
        self.max_responses = max_responses
        self.response_uppercase = response_uppercase
        self.utt_lowercase = utt_lowercase

    @staticmethod
    def get_dialog_act(sentence, classifier):
        if classifier == 'rule':
            dialog_act = RulebasedEstimator.predict(sentence)
        elif isinstance(classifier, pd.core.series):
            dialog_act = np.random.choice(list(classifier.index.values), 1, list(classifier.values))
        else:
            dialog_act = classifier.predict(sentence)
        return dialog_act

    # todo: accept a sentence and return the relevant slot and a value
    def get_matches(self, sentence):

        return {'area': 'south'}

    # todo: accept the restaurant data and preferences and return a filtered list based on preferences
    def query(self, preferences):
        queriedlist = ['macdonalds', 'kfc']
        return queriedlist

    # todo: out of a restaurant subset, return a random restaurant
    def get_random(self, restaurantSubset):
        restaurant = {'name': 'macdonalds',
                      'price': 'cheap',
                      'area': 'south',
                      'food': 'fast',
                      'phone': '04905903540',
                      'addr': 'my house',
                      'postcode': '2265dd'}
        return restaurant

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
            confirm = self.SENTENCES['confirmtype1'] + stake['type']
        if 'price_range' in stake:
            confirm = self.SENTENCES['confirmprice1'] + stake['price_range']
        if 'area' in stake:
            confirm = self.SENTENCES['confirmarea1'] + stake['area']

        return confirm

    def get_inform_sent(self, sentence):
        inform = 'information'
        return inform

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
            self.topic_at_stake = self.get_matches(sentence)
            new_state = 'confirm'
            response = self.get_confirm_sent()

        return response, new_state

    def state_request(self, sentence, act):
        response = ''
        new_state = 'request'

        if act == 'ack' or act == 'affirm':
            response = self.SENTENCES['ack1']

        if act == 'confirm' or act == 'reqalts' or act == 'inform' or act == 'request':
            self.topic_at_stake = self.get_matches(sentence)
            if self.info_per_utt == "all" and len(self.topic_at_stake) < 3:
                response = self.SENTENCES['reqall1']

            elif self.info_per_utt == "one" and len(self.topic_at_stake) > 1:
                response = self.SENTENCES['reqone1']

            elif self.confirmation_all:
                response = self.get_confirm_sent()
                new_state = 'confirm'
            else:
                # todo: update user preferences
                self.restaurantSet = self.query(self.user_preferences)
                if len(self.restaurantSet) < 1:
                    response = self.SENTENCES['noresults1']
                    self.user_preferences = self.EMPTY_PREFERENCES
                    new_state = 'request'

                if len(self.restaurantSet) < 5:
                    self.suggestion = self.get_random(self.restaurantSet)

                    name = self.suggestion['name']
                    price = self.suggestion['price']
                    area = self.suggestion['area']

                    response = f"What about {name}, its {price} and is in the {area}."
                    new_state = 'inform'

                if len(self.restaurantSet) >= 5:
                    response = self.get_request_sent()
                    new_state = 'request'

        if act == 'deny' or 'negate':
            response = self.SENTENCES['bye1']
            new_state = 'bye'

        if act == 'reqmore':
            response = self.SENTENCES['noise1']

        return response, new_state

    def state_confirm(self, sentence, act):
        response = ''
        new_state = 'confirm'

        if act == 'ack' or act == 'affirm':
            # todo: update user preferences
            self.user_preferences = []

            self.restaurantSet = self.query(self.user_preferences)
            if len(self.restaurantSet) < 1:
                response = self.SENTENCES['noresults1']
                new_state = 'confirm'

            if len(self.restaurantSet) < 5:

                self.suggestion = self.get_random(self.restaurantSet)

                name = self.suggestion['name']
                price = self.suggestion['price']
                area = self.suggestion['area']

                response = f"What about {name}, its {price} and is in the {area}."
                new_state = 'inform'

            if len(self.restaurantSet) >= 5:
                response = self.get_request_sent()
                new_state = 'request'

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
            # todo: check query
            if(True):
                response = self.SENTENCES['confirm1']
            if(False):
                new_state = 'confirm'
                response = self.get_confirm_sent(sentence)

        if act == 'deny' or 'negate' or 'reqmore':
            self.suggestion = self.get_random(self.restaurantSet)

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

        return self.response

    def start_conversation(self):

        print(self.SENTENCES['hello1'])
        sentence = input()
        while self.state != 'bye':
            print(self.get_next_sentence(sentence))
            sentence = input()

        return self.SENTENCES['ended1']


convo = Conversation(classifier='rule')
convo.start_conversation()








