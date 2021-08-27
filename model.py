from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from pattern.en import conjugate, PRESENT
import speech_recognition as sr 
import pyttsx3 
from vosk import Model, KaldiRecognizer
import os
import pyaudio
import json
import sys
from moviepy.editor import *

#our dataset
dictionary_words=['above','accept','accident','action','address','advice','aeroplane','after','air','all','allow','always',
        'ambulance','angry','animal','apple','arm','art','autorickshaw'
        'baby','ball','balloon','banana','bank','bat','bathroom','bed','behind','big','bird','birthday','black','breathe','buy',
        'cake','calculator','call','camera','car','cash','cat','chocolate','clap','cloth','computer','cricket','cry',
        'dance','date','deaf','diesel','discount','diwali','dog','drink',
        'education','egg','elephant','enjoy','eraser','exercise','eye',
        'face','fan','festival','film','fire','fish','football','free','friend','fruit',
        'ganapathi','girl','give','glass','green',
        'health','hear','heart','heat','heavy','helicopter','help','her','herself','hide','high','him','hindi','his','hold',
        'home','horse','hospital','how','hungry','husband',
        'i','identification','illegal','immediate','import','important','in','inch','income','increase','independent','index',
        'india','individual','infinity','information','initial','injection','input','installation','insult','interest',
        'introduce','inverse','iron','item','it','itself',
        'jail','jealous','job','join','juice','jump','june','junior','justice',
        'kanpur','keep','key','kick','kill','king','kiss','kite','know','knowledge',
        'labour','land','late','laugh','lazy','learn','leave','left','leg','less','letter','level','library','light','like',
        'lip','liquid','list','litre','little','local','lock','long','longitude','loose','lose','loud','love','lunch',
        'magic','magnet','mail','major','man','mango','mark','maximum','me','meet','money','month','more','music','myself',
        'natural','neck','need','neighbour','new','next','night','noon','nose','nosie','note','notice','now',
        'object','october','of','office','old','on','only','open','our','out','output','overdue'
        'Page','Pain','Pant','Paper','Pay','Percentage','Perfect','Play','Poor','Positive','Power','Practice','Process',
        'Product','Put',
        'Quailty','Question','Quit',
        'Read','Refund','Reject','Remove','Result','Return','Reward','Right','Risk',
        'She','Sad','Salt','Stop','Same','Solve','Sorry','Satisfied','Strong','Student','Study','Smile','Small','Self','Sister',
        'Transfer','Talk','Travel','Tight','Target','Time','Taste','Then', 'To','Try','Tomorrow','Total','Things','This',
        'Temporary','Trust','Table',
        'Up','Us',
        'Vegetables','Vehicles','Village','Vision','Visit','Volume',
        'Walk','Way','We','Weak','Who','Why','Win','Wish','Wrong',
        
        'Year','Yesterday','You','Your',
        'Zebra','Zone']
dictionary_words = [w.lower() for w in dictionary_words]

syn_dic={'above':['up','aloft','supra','more than','exceeding','over','upward'],
          'accept':['admit','concede','confess','recognize','avow','have','receive'],          
          'accident':['disaster', 'misadvanture','mishap','misfortune','misadventure','mischance','tragedy','trouble',
                      'casualty'],
          'action':['act','activity','gesture',],
          'address':['track','footprint','location','abode','domicile','dwelling','headquarters','house','lodging'],
          'advice' :['counsel','rede','consultation','guidance','advising','counseling','suggestion','opinion'], 
          'aeroplane':['plane','Airplane','flight'],  
          'after':['later','after this/that','subsequently','next','thereafter'], 
          'air':['wind','ventilate','breeze'],
          'all':['every','total','complete','fully','entirely','totally','wholly','thoroughly'],
          'allow':['permit','assent','grant','approve','permission','allowed','allowable'],
          'always':['perennially','consistently','regularly'],
          'angry':['annoyed', 'disgruntled', 'shirty','furious','displeased','fiery','fuming','irascible','ireful','wroth'],
          'animal':['cattle','beast','livestock','brute','fauna','creature'],
          'arm':['hand'],
          'art':['skill','mastership'],
          'baby':['infant','child','kiddy','kid','infante','babe','toddler'],
          'bat':['rearmouse'], 
          'bathroom':['Bagnio','lavatory','restroom','sauna','shower'], 
          'bed':['bedding','bedstead','cot','doss','bunk'],
          'big':['large','huge','large','sizeable','massive','gigantic','giant','stupendous','oversized','capacious','commodious'
                 ,'spacious'],
          'bird' :['birdie'], 
          'black':['dusky'],
          'breath':['inhale','heave','respire','inhalation'],
          'buy':['purchase','procure','bought'],
          'cake':['muffin','scone','pancake','cake'],   
          'call':['Invoke','epiclesis','Invocation','called','calling'],
          'camera':['camcorder'],
          'car':['roadster','wagon','motor','cabriolet','jeep','pickup','van'],
          'cat':['kitty','kitten','pussy','heck'],
          'chocolate':['candy','confection','dessert'],
          'clap':['applause','applaud'],
          'cloth':['fabric','cloths','weft','textile','clothes',],
          'computer':['desktop','laptop','minicomputer','microcomputer'],
          'cry':['blubber','weep','mourn','lament','bemoan'],
          'Dance':['dancing','shindig','orchestics'],
          'Deaf':['deafness'],
          'Discount':['exemption','rebate','concession','discounts','uncos'],
          'Diwali':['deepawali'],
          'Dog':['doggy','doggie','pooch','retriever','pup','puppy'],
          'Drink':['drinks','tipple'],
          'E-mail':['email','E-message'],
          'Education':['teaching','schooling'],
          'Egg':['testicle'],
          'Enjoy':['revel','delight','enjoyment','enjoyable','delightful','pleasurable'],
          'Eraser':['rubber','caoutchouc'],
          'Exercise':['yoga','workout','practice','exercising'],
          'Eye':['glimmers'],
          'Face':['countenance','visage'],
          'Fan':['ventilator', 'spiracle'],
          'Festival':['function', 'ceremony','celebration','fest','gala', 'festivities','festive','fete','fiesta'],
          'Film':['movies'],
          'Fire':['flame', 'blaze','bonfire','heat','conflagration'],
          'Football':['soccor'],
          'Free':['liberated','released','unfettered','immune','complimentary'],
          'Friend':['buddy','classmate','pal','mate','comrade','companion','fellow','ally','Mithras'],
          'Games':['play','sport'],
          'Ganapathi':['ganesh','gauriputra','bappa'],
          'Girl':['lady','schoolgirl','she','teenager','girlchild','gal','lass','wench','damsel'],
          'Give':['grant','pass', 'allow','render','entrust', 'cede', 'hand over', 'assign', 'deliver','provide'],
          'Glass':['glasswork','sandblast'],
          'Green':['greenish','viridescent'],
          'Health':['fitness','healthiness'],
          'Hear':['listen','hearken','hark','overhear'],
          'Heart':['cardia'],
          'Heat':['summer','caloric','warmth','heating','heated','superheat'],
          'Heavy':['bulky','cumbersome','cumbrous','overweight','heavily','heavyweight'],
          'Helicopter':['chopper'],
          'Help':['helping','support','helped','redound','bestead','assist'],
          'Her':['hers'],
          'Herself':['self','yourself','himself','oneself','yourselves'],
          'Hide':['conceal','stash','hideous'],
          'High':['advanced','towering'],
          'His':['her','hers'],
          'Hold':['grip','catch','clasp','clutch', 'seizing'],
          'Home':['residence','residency','habitation','accommodation','house','premises', 'dwelling','mansion'],
          'Horse':['steed','equine','destrier'],
          'Hospital':['asylum','clinic'],
          'Hungry':['starve','unfed','peckish','esurient'],
          'i':['me','mine'],
          'identification':['identity','id','pehchan','recognisance','recognition','recognizance','memento','souvenir','spotting'
                            ,'badge','naming',],
          'illegal':['lawless','outlawed','unauthorized','illicit','verboten','illegitimate','unlawful','outrule','wrongful',
                     'prohibited','bootlegged','malfeasance','misfeasance','crime','unethical ','unconstitutional',
                     'actionable','illegally','unauthorised','unlicenced'],
          'immediate':['instant','now','quick','urgent','instantaneous','immediately','instantaneously','immediacy',
                       'instantaneity','imminent'],
          'import':['convey','imported','importer','shipment'],
          'important':['considerable','significant','essential','crucial','foremost','valuable','necessary','mattering',
                       'pivotal','vital','major','urgent','useful','precious'],
          'in':['inwards','throughout','within','among','inside','toward'],
          'income':['revenue','earnings','salary','livelihood'],
          'increase':['boost','escalation','grow','elevate','expansion','addition','groundswell','accelerate','increased',
                      'incremental','extension','growth','hike','amplify','escalate','propagation','increasing','raise',
                      'increment'],
          'independent':['independence','individualistic','autonomous','autonomy','freelancer','independant','indepedent',
                         'individual'],
          'index':['glossary','token','pointer','indices','list','record','catalog','table','indexes','listing','appendix',
                  'tabulate'],
          'india':['bharat','hindustan','indian'],
          'individual':['alone','singular','personal','own','self','particular','be','lone','personalized','only','single',
                       'specific','independent','individuality','individually','individualized'],
          'infinity':['immeasurableness','extent','ellipse','infiniteness','unlimitedness','infinitude','unboundedness',
                     'infinite','endlessness'],
          'information':['message','data','info','detail'],
          'initial':['introductory','primary','basic','beginning','incipient','initially','early','start'],
          'injection':['dose','vaccination','inject','jab','inoculation','injectant'],
          'input':['enter','insert'],
          'installation':['installing','install','fitting','establishment','establishment','emplacement','enthronement',
                         'placing','setup','installed'],
          "insult":['snub','disgrace','offense','dishonor','shame','affront','defamation','contempt','mortification',
                   'disdain','reproach','scorn','spurn','mock','abasement','Derogation','Dishonour','disrespectfulness',
                    'humiliations','Inappreciation'],
          'introduce':['introduced','familiarize'],
          'inverse':['inverted','inverted','reversed','flipped','invert'],
          'item':['thing','object','artifact','product','items'],
          'it':['this','he','she'],
          'jail':['prison','imprisonment','gaol','lockup','slam','cell','dungeon'],
          'jealous':['envious','spiteful','rancorous','resentful','venomous','insecure','envy'],
          'job':['task','deed','doing','labor','profession','duty'],
          'join':['connect','flock','add','link','concatenate','joint','affiliate','clip','append','subjoin'],
          'juice':['liqur','extract','taille','sap','syrup'],
          'jump':['plunge','sally','leap','lope','leaping','bounce','jumped','hop'],
          'junior':['younger','lower'],
          'justice':['right','syllogism','rectitude','sconce','fairness','rightful'],
          'keep':['lay','put','hold','maintain','place','remain'],
          'kill':['murder','killing','assassination','slaughter','manslaughter','immolate','slay','behead'],
          'king':['monarch','raja','ruler','prince','earl','emperor','sovereign','ruler','majesty'],
          'labour':['labor','exertion','effort','toil','fatigue'],
          'land':['ground','soil','field','plot'],
          'late':['delayed'],
          'laugh':['laughter','jibe','jest','hoax','chortle','giggle'],
          'lazy':['sluggish','lingering','slothful','idle','unhurried','slack','dull','inactive','inert','leisurely','sluggard'],
          'leave':['renounce','cessation','letup','retraction','retractation','departure','exit','depart'],
          'leg':['foot','shank'],
          'less':['lower','short','lesser','shorter','fewer'],
          'light':['illumination','lightness','luminosity','radiance','lamp','shine','daylight'],
          'like':['identical','similar','alike'],
          'liquid':['fluid','liquefied'],
          'list':['liquefied','index'],
          'little':['some','slight','minor'],
          'lock':['locking','padlock','lockage','hasp','latch'],
          'long':['tall','lengthy','prolonged','protracted','high'],
          'loose':['easy','lax','relaxed'],
          'lose':['miss','squander','mislay','drop'],
          'loud':['noisy','louder','loudly','loudness','deafening'],
          'love':['affection','dearness','mash','romance','amor','adore','truelove'],
          'magic':['mystical','enchanted','magical','juggling','incantation','paternoster','voodoo'],
          'magnet':['loadstone','lodestone'],
          'major':['main','crucial','vital','considerable','primary','dominant'],
          'mark':['trace','notch','trail'],
          'maximum':['most','utmost','maximal','supreme'],
          'meet':['commingle','mingle','meeting'],
          'money':['wealth','specie','dinero'],
          'natural':['naturalistic','inartificial'],
          'neck':['throat','larynx','crag'],
          'need':['necessity','requirement','needfulness','needs','exigency','require','necessary','necessitate'],
          'neighbor':['neighboring','vicinal','neighboring'],
          'new':['latest','newly','newness'],
          'next':['forward','forthcoming','ensuing','subsequent'],
          'noon':['midday','noonday','noontime'],
          'nose':['conk','snitch'],
          'nosie':['din','clamor','pother','babel'],
          'object':['commodity','thing'],
          'office':['workplace','workroom'],
          'Page':['prshth','sheet','folio','leaf','recto'],
          'Pain':['suffering','anguish','soreness','ache','agony','discomfort','torment'],
          'Paper':['document', 'pasteboard','notepaper'],
          'Pay':['salary','wage','wages','compensate','income'],
          'Percentage':['rates','proportion','ratio'],
          'Perfect':['good','masterly','excellent','foolproof','ideal','impeccable','splendid','superb'],
          'Play':['sports','sporting','fun','pastime'],
          'Poor':['beggarly','penniless','penurious','destitute','impoverished','needy','underprivileged'],
          'Power':['authority','strength','force','energy','potency','vigor','capability','capacity','potential'],
          'Practice':['exercises', 'exercise', 'training', 'drill'],
          'Process':['procedure','mechanism','operation','procedure','proceeding','step','technique'],
          'Product':['commodity','merchandise'],
          'Read':['reading','study','learn'],
          'Refund':['withdrawal','refundment','refunds','compensate'],
          'Reject':['nay','rejection','denial','negation','elimination','exclusion','repudiation'],
          'Remove':['recapture','discard','disseverance','delete','discard','discharge','dismiss','eliminate','expel'],
          'Result':['outcome','consequence','outgrowth','effect','peroration'],
          'Return':['comeback','repayment','regress','recoil','turn'],
          'Reward':['prize','bounty','regalia','pewter','payoff','honor'],
          'right':['correct','proper','reasonable','advisable','pertinent'],
          'Risk':['hazard','peril','riskiness','endanger','imperil','jeopardize','thret'],
          'zone':['area','sector','section','region','territory','district','province'],
          'year': ['annual','annually','annum','yr'],
          'yesterday':['past','foretime','recently'],
         'you':['yourself','thee'],
         'way':['path','route','walkway','aisle'],
         'walk':['stroll','saunter','amble','trudge','tread','footslog','traipse','perambulate','excursion',],
         'we':['us','ourselves','ourself'],
         'weak':['feeble','puny','fragile','delicate','weakly','infirm','debilitated','incapacitated','enervated','tired','exhausted'
         ,'diminutive','lean','svelte','sleazy','powerless'],
         'who':['whom','whose'],
         'why':['reason'],
         'win':['victory','triumph','succeed','conquer','vanquish','outvie','overcrow','overbear','winnings','won','winning'],
         'wish':['desire','hope','prayer','dream','want','longing','aspiration','aspire','wishing'],
         'wrong':['misfigured','misconstrued','incorrect','immoral','false','error','inaccurate','erroneous'],
         'vegetable':['legumes','vegies','veg','sabji'],
         'vehicle':['transportation','transport','carrier','car','lorry','motorcycle','conveyance','carriage'],
         'village':['hamlet','pueblo','dorp','thorp','townlet','town'],
         'vision':['sight','foresight','eyesight','insight','perspective'],
          'visit':['visitation','tour','travel'],
         'volume':['voluminous'],
         'transfer':['transmit','convey','consign','shifting','transference','transposition','displacement','transferal','devolution'
               ,'handover'],
         'talk':['chatting','gossiping','talking','chat','parley','conversation','consort',],
         'travel':['journey','tour','trip','excursion','peregrination','wayfaring','trek'], 
         'target':['goal','aim','objective','purpose'],
         'taste':['relish','savor','savour','flavour','fruition','gustation','tasty'],
         'total':['thorough','overall','entire','exhaustive','all','totality','entirety','everything'],
         'this':['it','he','she'],
          'try':['effort','endeavor','attempt'],
         'temporary':['makeshift','streaky','changeable','unstable'],
         'trust':['belief','faith','believe','rely'],
         'sad':['unhappy','sorrowful','miserable','tragi','gloomy','depressed','glum','joyless','grieving','grieved'],
         'salt':['sal','saline','salted','salty'],      
         'same':['common','usual','generic','Equal','selfsame','identical','similar','synonymic','alike','similarly','identic',
                'equality'],
         'sorry':['apology','regret','apologetic','regretful','remorseful'],
         'satisfied':['contented','acquiescent'],
         'strong':['powerful','mighty','muscular','firm','strudy','oaky','massy','sinewy','robust'],
         'student':['pupil','disciple','schoolboy','schoolchild','schoolgirl'],
         'study':['perusal','learning','schooling'],
         'smile':['grin','laugh','giggle'],
         'small':['miniature','mini','little','short','minor','tiny'],
         'self':['oneself','mine'],
         'up':['above','aloft','supra'],
         'us':['we','ourself']
    }


def sentence(inpu):
    if inpu == 'audio_online':
        r = sr.Recognizer()  

        with sr.Microphone() as source:
            print("Talk")
            audio_text = r.listen(source)
            print("Time over, thanks")
            try:
                inputString= r.recognize_google(audio_text).lower()
            except:
                 print("Sorry, I did not get that")

    elif inpu == 'audio_offline':

        if not os.path.exists("model"):
            print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit (1)

        model = Model("model")
        rec = KaldiRecognizer(model, 16000)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        k=0

        while True:
            data = stream.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                print("Time over, Thanks")
                inputString = json.loads(rec.Result())['text'].lower()
                k=1
            else:
                if(k==1):
                    break
                if(k==0):
                    print("I'm listening, carry on")
                    k=2

    else:
        print("Enter the text")
        inputString = input().lower()
    
    print("Your input is -", inputString)
    
    return inputString

print("If you want accuracy and ready to give a few time then input more otherwise less")
tim = input().lower() 

if (tim=="more"):
    from pycontractions import Contractions
    cont = Contractions(api_key="glove-twitter-100")
else:
    import contractions

print("If you want to use the audio mode and you have a internet connection then enter audio_online")
print("If you want to use the audio mode and don't have a internet connection then enter audio_offline")
print("If you want to use text mode then enter text")
inpu=input()

inputString = sentence(inpu)

if(tim=="more"):
    inputString = list(cont.expand_texts([inputString], precise=True))[0]
else:
    inputString=contractions.fix(inputString)
    
tokenizer = RegexpTokenizer(r'\w+')
inputString = tokenizer.tokenize(inputString)


words_inter=['am', 'are', 'can', 'could', 'did', 'do', 'does', 'had', 'has', 'have', 'how', 'is', 'may','shall','should','were',
             'what','when', 'where','whether','which','who','whom','whose', 'why', 'will', 'would'] 
if inputString[0] in words_inter:
    start_word= inputString[0]
    inter=1
    inputString=inputString[1:]
else:
    inter=0

inputString=' '.join(inputString)

parser=CoreNLPParser(url='http://localhost:9000')
englishtree=[tree for tree in parser.parse(inputString.split())]
parsetree=englishtree[0]
pos_tags = parsetree.pos()
pos_tags_dic={}
for i in pos_tags:
    pos_tags_dic[i[0]] = i[1][0]
    
parenttree= ParentedTree.convert(parsetree) 
isltree=Tree('ROOT',[])
dic={}
for sub in parenttree.subtrees(): 
    dic[sub.treeposition()]=0
i=0

for sub in parenttree.subtrees():
    if(sub.label()=="NP" and dic[sub.treeposition()]==0 and dic[sub.parent().treeposition()]==0 and len(sub.leaves())==1):
        dic[sub.treeposition()]=1
        isltree.insert(i,sub)
        i=i+1
    if(sub.label()=="VP" or sub.label()=="PRP"):
        for sub2 in sub.subtrees():
            if((sub2.label()=="NP" or sub2.label()=='PRP')and dic[sub2.treeposition()]==0 and 
                    dic[sub2.parent().treeposition()]==0 and len(sub2.leaves())==1):
                dic[sub2.treeposition()]=1
                isltree.insert(i,sub2)
                i=i+1
    
for sub in parenttree.subtrees():
    for sub2 in sub.subtrees():
        if(len(sub2.leaves())==1 and dic[sub2.treeposition()]==0 and dic[sub2.parent().treeposition()]==0 and 
                         len(sub2.leaves())==1):
            dic[sub2.treeposition()]=1
            isltree.insert(i,sub2)
            i=i+1
words=isltree.leaves()
if(inter==1):
    words.append(start_word)
    
stop_words=['be','as','am','for','than','a','an','the','of', 'been', 'being','and','but','if','or','because','as','while','by',
            'for','such','own','so','than','too','very','just',"'s"]

def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def get_key(val): 
    for key, value in syn_dic.items(): 
        if val in value: 
            return key 
    return "key doesn't exist"

lemmatizer = WordNetLemmatizer()
lemmatized_words=[]
for w in words:
    try:
        tag= pos_tags_dic[w]
    except:
        tag = nltk.pos_tag([w])[0][1][0].upper()
    try:
        if (tag == 'V'):
            w = conjugate(verb=w,tense=PRESENT)
    except:
        w=w
    lemmatized_words.append(lemmatizer.lemmatize(w,get_wordnet_pos(tag)))
islsentence = ""

previous_word=""

final_output_array=[]
for w in lemmatized_words:
    if w == previous_word:
        continue
    else:
        previous_word=w
        
    if w not in stop_words:
        if w in dictionary_words:
            islsentence+=w
            islsentence+=" "
            final_output_array.append(w)
        else:
            key=get_key(w)
            if(key=="key doesn't exist"):
                continue
            else:
                islsentence+=key
                islsentence+=" "
                final_output_array.append(key)


clips=[]
root='C:\Shubh\Study MAterial\deaf_dataset\ISL'


displayed_words=""

for words in final_output_array:
    capital = words.capitalize()
    first_letter= words[0].upper()
    try:
        filePath = os.path.join(root, first_letter,capital+'.mp4')
        clips.append(VideoFileClip(filePath))
        displayed_words+=words
        displayed_words+=" "
    except:
        continue

        
final = concatenate_videoclips(clips)
final.resize(width=480)
print("The words displayed are-")
print(displayed_words)
final.ipython_display(width = 480)
