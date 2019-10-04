### FUNCTIONS FOR INTRODUCTION

### user preference class ###

# conversation class
class conversation:
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
        'ended1': 'The dialog has ended.'
    }

    def __init__(self):
        self.state = 'hello'
        self.response = 'nothing'
        self.userPreferences = {"food": [], "price_range": [], "area": []}
        self.restaurantSet = []
        self.suggestion = {}
        self.infoGiven = False
        self.topic_at_stake = {}

    #todo: work in progress
    def getdialogact(self, sentence):
        dialogact = 'inform'
        return dialogact

    #todo: work in progress
    def getMatches(self, sentence, key='area'):

        return({'area': 'south'})

    #todo: work in progress
    def query(self, data, preferences):
        queriedlist = ['macdonalds', 'kfc']
        return(queriedlist)

    #todo: work in progress
    def getRandom(self, restaurantSubset):
        restaurant = {'name': 'macdonalds',
                      'price': 'cheap',
                      'area': 'south',
                      'food': 'fast',
                      'phone': '04905903540',
                      'addr': 'my house',
                      'postcode': '2265dd'}
        return(restaurant)


    def get_request_sent(self, sentence):
        request = 'what do you want?'
        return request

    def get_confirm_sent(self, sentence):
        confirm = "so u want cheese"
        return confirm

    def get_suggest_sent(self, sentence):
        suggest = 'suggestion'
        return suggest

    def get_inform_sent(self, sentence):
        inform = 'information'
        return inform


    def state_hello(self, sentence, act):
        response = ''
        new_state = 'hello'

        if(act == 'ack' or act == 'affirm'):
            new_state = 'request'
            response = self.SENTENCES['ack1']

        if(act == 'confirm'):
            new_state = 'request'
            response = self.SENTENCES['request1']

        if(act == 'deny'):
            new_state = 'bye'
            response = self.SENTENCES['bye2']

        if(act == 'reqmore' or act == 'request'):
            response = self.SENTENCES['noise1']

        if(act == 'inform' or act == 'deny' or act == 'reqalts'):
            new_state = 'confirm'
            response = self.get_confirm_sent(sentence)

        return(response, new_state)

    def state_request(self, sentence, act):
        response = ''
        new_state = 'request'

        if (act == 'ack' or act == 'affirm'):
            response = self.SENTENCES['ack1']

        if (act == 'confirm' or act == 'reqalts' or act == 'inform' or act == 'request'):
            response = self.get_confirm_sent(sentence)
            new_state = 'confirm'

        if (act == 'deny'):
            response = self.SENTENCES['bye1']
            new_state = 'bye'

        if(act == 'reqmore'):
            response = self.SENTENCES['noise1']

        return(response, new_state)

    def state_confirm(self, sentence, act):
        response = ''
        new_state = 'confirm'

        if(act == 'ack' or act == 'affirm'):
            # update prefs
            # update subset
            # update at stake

            if(len(self.restaurantSet) < 1):
                response = self.SENTENCES['noresults1']
                new_state = 'confirm'

            if (len(self.restaurantSet) < 5):

                self.suggestion = self.getRandom(self.restaurantSet)

                name = self.suggestion['name']
                price = self.suggestion['price']
                area = self.suggestion['area']

                response = f"What about {name}, its {price} and is in the {area}."
                new_state = 'inform'

            if (len(self.restaurantSet) >= 5):
                response = self.get_request_sent(sentence)
                new_state = 'request'

        if(act == 'confirm' or act == 'inform'):
            response = self.get_confirm_sent(sentence)
            new_state = 'confirm'

        if(act == 'reqalts'):
            response = self.SENTENCES['softreset1']
            new_state = 'request'

        if(act == 'deny' or 'negate'):
            # clear at stake slots
            new_state = 'request'
            response = self.SENTENCES['tryagain1']

        return(response, new_state)

    def state_inform(self, sentence, act):
        response = ''
        new_state = 'inform'

        if(act == 'inform'):
            response = self.get_confirm_sent(sentence)
            new_state = 'confirm'

        if(act == 'confirm'):
            # todo: check query
            if(True):
                response = self.SENTENCES['confirm1']
            if(False):
                new_state = 'confirm'
                response = self.get_confirm_sent(sentence)

        if(act == 'deny' or 'negate' or 'reqmore'):
            self.suggestion = self.getRandom(self.restaurantSet)

            name = self.suggestion['name']
            price = self.suggestion['price']
            area =  self.suggestion['area']

            response = f"What about {name}, its {price} and is in the {area}."

        if(act == 'ack' or 'affirm'):
            if(self.infoGiven == False):
                response = self.get_inform_sent(sentence)

            if(self.infoGiven == True):
                response = 'goodbye'
                new_state = ''

        if(act == 'request'):
            response = self.get_inform_sent(sentence)

        if(act == 'inform'):
            new_state = 'confirm'

        return(response, new_state)

    def state_bye(self, sentence):
        response = self.SENTENCES['bye1']
        new_state = 'bye'

        return(response, new_state)

    def getNextSentence(self, sentence=None):

        switcher = {
            "hello": self.state_hello,
            "request": self.state_request,
            "confirm": self.state_confirm,
            "inform": self.state_inform,
            "bye": self.state_bye
        }

        # get dialog act
        act = self.getdialogact(sentence)

        # general responses
        if(act == 'hello'):
            new_state = 'request'
            response = self.SENTENCES['hello2']

        if (act == 'repeat'):
            return (self.SENTENCES['repeat1'] + self.response)

        if (act == 'null'):
            self.response = self.SENTENCES['noise1']
            return (self.response)

        if (act == 'bye'):
            return (self.SENTENCES['bye1'])

        if (act == 'restart'):
            self.__init__()
            return (self.SENTENCES['restart1'])

        if (act == 'thankyou'):
            return (self.SENTENCES['thankyou1'])

        func = switcher.get(self.state, lambda: "Invalid State")

        response, new_state = func(sentence, act)
        self.state = new_state
        self.response = response

        return(self.response)

    def start_conversation(self):

        print(self.SENTENCES['hello1'])
        sentence = input()
        while(self.state != 'bye'):
            print(self.getNextSentence(sentence))
            sentence = input()

        return self.SENTENCES['ended1']


convo = conversation()
convo.start_conversation()








