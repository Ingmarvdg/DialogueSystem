### FUNCTIONS FOR INTRODUCTION

### user preference class ###

# conversation class
class conversation:
    sentences = {
        "hello1": "Hello! Welcome to the Ambrosia restaurant system. What kind of restaurant are you looking for?",
        "hello2": "Hi there! You can search for restaurants by area, price range or food type. What would you like?",
        "empty1": "Sorry, no restaurants matching your criteria were found. Would you like to try something else?",
        "repeat1": "Oh sorry. I'm gonna repeat that.",
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
        "suggest1": "What about ",
        "confirm1": "Yes,that is correct."
        "deny1": "No."

    def __init__(self):
        self.phase = 'gathering'
        self.act = 'start'
        self.last_message = 'Nothing'
        self.userPreferences = {"food": [], "price_range": [], "area": []}
        self.query = []
        self.madeSuggestion = False
        self.infoGiven = False
        self.SENTENCES = {
        }

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

    def getNextSentence(self, restaurantSubset, sentence=None):
        SUGGESTLIMIT = 5
        SENTENCES = {
            "hello1": "Welcome "

        }

        # initial message
        if(self.phase == 'hello' and sentence is None):
            self.last_message = SENTENCES['hello1']
            return(self.last_message)

        # get dialog act
        self.act = self.getdialogact(sentence)

        # repeat message
        if(self.act == 'repeat'):
            return(SENTENCES["repeat1"] + self.last_message)

        # noise message
        if(self.act == 'null'):
            self.last_message = SENTENCES['noise1']
            return(self.last_message)

        # bye message
        if(self.act == 'bye'):
            return(SENTENCES['bye1'])

        # restart message
        if(self.act == 'restart'):
            self.act = 'start'
            self.phase = 'hello'
            self.last_message = None
            return(SENTENCES['restart1'])

        # thank you message
        if(self.act == 'thankyou'):
            return(SENTENCES['thankyou1'])

        # hello phase
        if(self.phase =='Hello'):
            if(self.act == 'hello'):
                return(SENTENCES["hello2"])
            if(self.act == 'ack' or self.act == 'affirm'):
                return(SENTENCES['ack1'])
            if(self.act == 'confirm'):
                return(SENTENCES['request1'])
            if(self.act == 'deny'):
                return(SENTENCES['bye2'])
            if(self.act == 'reqmore' or self.act == 'request'):
                return(SENTENCES['noise1'])
            if(self.act == 'inform' or self.act == 'deny' or self.act == 'reqalts'):
                self.phase = 'gathering'

        # gathering phase
        if(self.phase =='gathering'):
            # todo: update preferences based on the sentence
            # todo: update subset

            if(restaurantSubset < 1):
                self.userPreferences = {"food": [], "price_range": [], "area": []}
                return(self.SENTENCES['empty1'])

            if(restaurantSubset < SUGGESTLIMIT):
                self.phase = 'suggestions'

            if(restaurantSubset > SUGGESTLIMIT):
                if(self.act == 'hello' or self.act =='reqmore'):
                    return(self.SENTENCES['noise1'])

                if(self.act == 'inform'):
                    if (self.userPreferences['area'] == []):
                        return (self.SENTENCES['area1'])
                    if (self.userPreferences['price_range'] == []):
                        return (self.SENTENCES['pricerange1'])
                    if (self.userPreferences['type'] == []):
                        return (self.SENTENCES['type1'])
                    else:
                        self.phase = 'suggestions'

                if(self.act == 'ack' or self.act == 'affirm' or self.act == 'negate' or self.act == 'reqmore'):
                    return(self.last_message)

                if(self.act =='confirm'):
                    self.phase = 'confirm'

                if(self.act =='deny'):
                    self.phase = 'confirm'

                if(self.act == 'reqalts'):
                    if (self.userPreferences['area'] == []):
                        return (self.SENTENCES['area1'])
                    if (self.userPreferences['price_range'] == []):
                        return (self.SENTENCES['pricerange1'])
                    if (self.userPreferences['type'] == []):
                        return (self.SENTENCES['type1'])
                    else:
                        self.phase = 'suggestions'

        if(self.phase == 'suggestions'):
            if(self.madeSuggestion == False
                    or self.act == 'deny'
                    or self.act == 'negate'
                    or self.act == 'reqalts'
                    or self.act == 'reqmore'):

                self.suggestion = self.getRandom(restaurantSubset)

                return(self.SENTENCES['suggest1']
                       + self.suggestion['name']
                       + " it's"
                       + self.suggestion['price']
                       + " and its in the "
                       + self.suggestion['area'] )

            if(self.act == 'ack' or self.act == 'affirm'):
                if(self.infoGiven == False):
                    return(self.SENTENCES['inform1']
                           + self.suggestion['name']
                           + self.suggestion['addr']
                           + self.suggestion['postcode']
                           + self.suggestion['phone'])
                if(self.infoGiven == True):
                    self.phase = 'goodbye'

            if(self.act == 'confirm'):
                # get match and check
                if(True):
                    return(SENTENCES['confirm1'])
                if(False):
                    return(SENTENCES['deny1'])

            if(self.act == 'hello'):
                return(SENTENCES['noise1'])

            if(self.act == 'inform'):
                self.phase = 'confirm'

            if(self.act == 'request'):
                # get match and return info
                return("this doesnt work yet")


        if(self.phase == 'confirm'):
            return('ok')


        if(self.phase == 'goodbye'):
            return('bye')




convo = conversation()
sent = 'Hello I want to have italian food in center pls'
print(convo.getNextSentence(sent))







