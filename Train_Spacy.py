# coding=utf-8
from __future__ import unicode_literals, print_function
import json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger
import unicodedata
import sys
import textblob
try:
    unicode
except:
    unicode = str


def train_ner(nlp, train_data, entity_types):
    # Add new words to vocab.
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]

    # Train NER.
    ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
    for itn in range(20):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            ner.update(doc, gold)
    return ner

def save_model(ner, model_dir):
    model_dir = pathlib.Path('D:\\PYTHON\\result\\POSITION')
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    with (model_dir / 'config.json').open('wb') as file_:
        data = json.dumps(ner.cfg)
        if isinstance(data, unicode):
            data = data.encode('utf8')
        file_.write(data)
    ner.model.dump(str(model_dir / 'model'))
    if not (model_dir / 'vocab').exists():
        (model_dir / 'vocab').mkdir()
    ner.vocab.dump(str(model_dir / 'vocab' / 'lexemes.bin'))
    with (model_dir / 'vocab' / 'strings.json').open('w', encoding='utf8') as file_:
        ner.vocab.strings.dump(file_)


def main(model_dir=None):
    nlp = spacy.load('en', parser=False, entity=False, add_vectors=False)

    # v1.1.2 onwards
    if nlp.tagger is None:
        print('---- WARNING ----')
        print('Data directory not found')
        print('please run: `python -m spacy.en.download --force all` for better performance')
        print('Using feature templates for tagging')
        print('-----------------')
        nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

    train_data = [
        (
            u"Masheesh Ikram\nLEAD SOFTWARE ENGINEER\nSupply Chain | Research & Development\nIFS R&D International, \nNo 501, Galle Road, Colombo 06, SRI LANKA\nTel +94 (0) 11 2364 400. Fax +94 (0) 11 2364401. Mobile +94 (0) 779050954\nmasheesh.ikram@ifsworld.com | www.IFSWORLD.com \nIFS World Operations AB is a limited liability company registered in Sweden. \nCorporate identity number: 556040-6042. \nRegistered office: Teknikringen 5, Box 1545, SE-581 15 Linköping.",
            [(len(u'Masheesh Ikram\nLEAD SOFTWARE ENGINEER\nSupply Chain | Research & Development\nIFS R&D International, \nNo 501, Galle Road, Colombo 06, SRI LANKA\nTel +94 (0) 11 2364 400. Fax '),len(u'Masheesh Ikram\nLEAD SOFTWARE ENGINEER\nSupply Chain | Research & Development\nIFS R&D International, \nNo 501, Galle Road, Colombo 06, SRI LANKA\nTel +94 (0) 11 2364 400. Fax +94 (0) 11 2364401'),'FAX'),(15,37,'POS'),
        (146, 165, 'TEL'),(198, 215, 'MOB'),(103,141,'add')]),
        (
            u"Asanka Gallege\nSecretary | IFS Welfare\n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 11 236 4400 (ext. 1722). Fax +94 11 236 4401. Mobile +94 71 563 9556\nasanka.gallege@ifsworld.com | www.IFSWORLD.com \nIFS World Operations AB is a limited liability company registered in Sweden. \nCorporate identity number: 556040-6042. \nRegistered office: Teknikringen 5, Box 1545, SE-581 15 Linköping.",
            [(len(u'Asanka Gallege\n'), len(u'Asanka Gallege\nSecretary'), 'POS'), (len(u'Asanka Gallege\nSecretary | IFS Welfare\n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 11 236 4400 (ext. 1722). Fax '),
            len( u'Asanka Gallege\nSecretary | IFS Welfare\n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 11 236 4400 (ext. 1722). Fax +94 11 236 4401'),'FAX'), (84, 99, 'TEL'),(141, 156, 'MOB'),(39,79,'add')]),

        (
            u"David Anderson\nEmail: donato@example.com\nChief Executive Officer\nOffice  800-555-5555 \nBroadlook Technologies	\nCell :  414-555-5555 \n21140 Capitol Drive\nFax : 262-754-8081\nPewaukee WI 53072\nBlog www.idanato.com\nhttp://www.broadlook.com",
            [(len(u'David Anderson\nEmail: donato@example.com\n'),
              len(u'David Anderson\nEmail: donato@example.com\nChief Executive Officer'), 'POS'), (len(
                u'David Anderson\nEmail: donato@example.com\nChief Executive Officer\nOffice  800-555-5555 \nBroadlook Technologies	\nCell :  414-555-5555 \n21140 Capitol Drive\nFax : '),
                                                                                                   len(
                                                                                                       u'David Anderson\nEmail: donato@example.com\nChief Executive Officer\nOffice  800-555-5555 \nBroadlook Technologies	\nCell :  414-555-5555 \n21140 Capitol Drive\nFax : 262-754-8081'),
                                                                                                   'FAX'),
             (73, 85, 'TEL'),
             (119, 131, 'MOB')]),
        (
            u"Valerie Richardson \nAccountant\n2906 N. Glenwood Terrace, Atlanta, GA 30310\n(404) 555-0789\nValerie.Richardson@yahoo.com\n501, Galle Road, , Colombo 06,  SRI LANKA\nTel +94 11 236 44 00. Fax +94 11 236 44 01\nchathuri.gamage@ifsworld.com | www.IFSWORLD.com ",
            [(len(u'Valerie Richardson \n'), len(u'Valerie Richardson \nAccountant'), 'POS'), (len(
                u'Valerie Richardson \nAccountant\n2906 N. Glenwood Terrace, Atlanta, GA 30310\n(404) 555-0789\nValerie.Richardson@yahoo.com\n501, Galle Road, , Colombo 06,  SRI LANKA\nTel +94 11 236 44 00. Fax '),
                                                                                               len(
                                                                                                   u'Valerie Richardson \nAccountant\n2906 N. Glenwood Terrace, Atlanta, GA 30310\n(404) 555-0789\nValerie.Richardson@yahoo.com\n501, Galle Road, , Colombo 06,  SRI LANKA\nTel +94 11 236 44 00. Fax +94 11 236 44 01'),
                                                                                               'FAX'),
             (165, 181, 'TEL'),(119,160,'add')]),
        (
            u'Kandasamy Yogendirakumar (Yogi)\nMSc, MBCS, MIET | DIRECTOR IFS ACADEMY \n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 (0)112 364 440. Fax +94 (0)112 364 441. Mobile +94 (0)714 039 089 \nkandasamy.yogendirakumar@ifsworld.com|www.IFSWORLD.com \nIFS World Operations AB is a limited liability company registered in Sweden. \nCorporate identity number: 556040-6042. \nRegistered office: Teknikringen 5, Box 1545, SE-581 15 Linköping.',
            [(len(u'Kandasamy Yogendirakumar (Yogi)\nMSc, MBCS, MIET | '),
              len(u'Kandasamy Yogendirakumar (Yogi)\nMSc, MBCS, MIET | DIRECTOR'), 'POS'), (len(
                u'Kandasamy Yogendirakumar (Yogi)\nMSc, MBCS, MIET | DIRECTOR IFS ACADEMY \n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 (0)112 364 440. Fax '),
                                                                                            len(
                                                                                                u'Kandasamy Yogendirakumar (Yogi)\nMSc, MBCS, MIET | DIRECTOR IFS ACADEMY \n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 (0)112 364 440. Fax +94 (0)112 364 441'),
                                                                                            'FAX'), (117, 135, 'TEL'),
             (168, 186, 'MOB'),(72,112,'add')]),
        (u'I am an Engineer',
         [(len(u'I am an '), len(u'I am an Engineer'), 'POS')]),
        (u'I am an Lead Engineer as well as Software Engineer in IFS.',
         [(len(u'I am an '), len(u'I am an Lead Engineer'), 'POS'),
          (len(u'I am an Lead Engineer as well as '), len(u'I am an Lead Engineer as well as Software Engineer'),
           'POS')]),
        (u'JOHN JONES\nMARKETING MANAGER\n10-123 1/2 MAIN STREET NW\nMONTREAL QC  H3Z 2Y7\nCANADA ',
         [(len(u'JOHN JONES\n'), len(u'JOHN JONES\nMARKETING MANAGER'), 'POS')]),
        (u'Ms. FNS Wijayakulasooriya\nMechanical Engineer\nsuranji@ugc.ac.lk\n+94 11 2669652	+94 11 673663',
         [(len(u'Ms. FNS Wijayakulasooriya\n'), len(u'Ms. FNS Wijayakulasooriya\nMechanical Engineer'), 'POS'),
          (64, 78, 'TEL'), (79, 92, 'TEL')]),
        (
            u'Jane Smith\nPhotographer\nA  1905 Hill Road, MI\nP 333-55551   O 333-55552 \nM 555-77777 \nF 555-99999 \nE owner@company.com \nW http://www.company.com  W http://www.example.com\nSkype John2k15',
            [(len(u'Jane Smith\n'), len(u'Jane Smith\nPhotographer'), 'POS'), (
            len(u'Jane Smith\nPhotographer\nA  1905 Hill Road, MI\nP 333-55551   O 333-55552 \nM 555-77777 \nF '), len(
                u'Jane Smith\nPhotographer\nA  1905 Hill Road, MI\nP 333-55551   O 333-55552 \nM 555-77777 \nF 555-99999'),
            'FAX'), (48, 57, 'TEL'), (75, 84, 'MOB')]),
        (u'David Anderson\nSecretary',
         [(len(u'David Anderson\n'), len(u'David Anderson Secretary'), 'POS')]),
        (
            u'Asanka Gallege\nSecretary | IFS Welfare\n501, Galle Road, Colombo 06, SRI LANKA\nTel +94 11 236 4400 (ext. 1722). Fax +94 11 236 4401. Mobile +94 71 563 9556',
            [(len(u'Asanka Gallege\n'), len(u'Asanka Gallege\nSecretary'), 'POS'), (len(
                u'Asanka Gallege\nSecretary | IFS Welfare\n501, Galle Road, Colombo 06, SRI LANKA\nTel +94 11 236 4400 (ext. 1722). Fax '),
                                                                                    len(
                                                                                        u'Asanka Gallege\nSecretary | IFS Welfare\n501, Galle Road, Colombo 06, SRI LANKA\nTel +94 11 236 4400 (ext. 1722). Fax +94 11 236 4401'),
                                                                                    'FAX'), (82, 97, 'TEL'),
             (139, 154, 'MOB'),(39,77,'add')]),
        (
            u'Fredrik Vom\nGROUP SENIOR VICE PRESIDENT\nBusiness Development\nGullbergs Strandgata 15, SE-411 04 Goteborg,SWEDEN\nTel +46 31 726 3046. Fax +46 31726 3001. Mobile +46 733 453046\nfredrik.vom.hofe@ifsworld.com | www.IFSWORLD.com\nIFS World Operations AB is a limited liability company registered in Sweden.',
            [(len(u'Fredrik Vom\n'), len(u'Fredrik Vom\nGROUP SENIOR VICE PRESIDENT'), 'POS'), (137, 151, 'FAX'),
             (116, 131, 'TEL'), (160, 174, 'MOB'),(61,111,'add')]),
        (
            u'Dr. Ashok Padhye\nGeneral Physician\nA-205, Natasha Apartments\n2, Inner Ring Road\nDomlur\nBANGALORE - 560071\nKarnataka',
            [(len(u'Dr. Ashok Padhye\n'), len(u'Dr. Ashok Padhye\nGeneral Physician'), 'POS'),(61,105,'add')]),
        (
            u'Dr. Ashok Padhye\nGeneral Physician\nA-205, Natasha Apartments\n2, Inner Ring Road\nDomlur\nBANGALORE - 560071\nKarnataka',
            [(len(u'Dr. Ashok Padhye\n'), len(u'Dr. Ashok Padhye\nGeneral Physician'), 'POS')]),
        (
            u'Ms. DD Kariyawasam\nArchitect\ndamitha@ugc.ac.lk\nTelephone : +94 11 5329500\nExt : 9500\nFax : +94 11 2885358\nEmail : controller[at]immigration.gov.lk\n+94 11 2678731',
            [(len(u'Ms. DD Kariyawasam\n'), len(u'Ms. DD Kariyawasam\nArchitect'), 'POS'), (91, 105, 'FAX'),
             (59, 73, 'TEL')]),
        (
            u'At. Sr. Hiro Gordo-Globo\nTeacher\nSumo Informtica S.A.\nCalle 39 No 1540\nB1000TBU San Sebastian',
            [(len(u'At. Sr. Hiro Gordo-Globo\n'), len(u'At. Sr. Hiro Gordo-Globo\nTeacher'), 'POS'),(54,93,'add')]),
        (
            u'Mr. HD Rasika Karunarathna\nMathematical Technicians\n+94 11 2685758	+94 11 2691678\nshalika@ugc.ac.lk\nARGENTINA ',
            [(
             len(u'Mr. HD Rasika Karunarathna\n'), len(u'Mr. HD Rasika Karunarathna\nMathematical Technicians'), 'POS'),
             (52, 66, 'TEL'), (67, 81, 'TEL')]),
        (
            u'James Cameroon\nSurveying Technicians\nStockton Campus\n3601 Pacific Avenue, Stockton, California 95211\n209.946.2285',
            [(
                len(u'James Cameroon\n'), len(u'James Cameroon\nSurveying Technicians'), 'POS'),(53,100,'add')]),
        (
            u'Mr. A. L. Perera\nGeographers\n201 Silkhouse Street\nKANDY\n20000\nSRI LANKA',
            [(
                len(u'Mr. A. L. Perera\n'), len(u'Mr. A. L. Perera\nGeographers'), 'POS'),(29,71,'add')]),
        (
            u'Sr. Francisco Ansó García\nAir Crew Officers\nPaseo de la Castellana, 185, 5ºB\n29001 Madrid\nMadrid.Tel: +94 055 854 12 13 TEL : +94 0225647845',
            [(
                len(u'Sr. Francisco Ansó García\n'), len(u'Sr. Francisco Ansó García\nAir Crew Officers'), 'POS'),(102,119,'TEL'),(126,140,'TEL'),(44,89,'add')]),
        (
            u'Valerie Richardson\n2906 N. Glenwood Terrace, Atlanta, GA 30310\n(404) 555-0789\nBusiness Operations Specialists\nValerie.Richardson@yahoo.com\n501, Galle Road, , Colombo 06,   SRI LANKA\nTel +94 11 236 44 00. Fax +94 11 236 44 01 \nchathuri.gamage@ifsworld.com | www.IFSWORLD.com ',
            [(
                len(u'Valerie Richardson\n2906 N. Glenwood Terrace, Atlanta, GA 30310\n(404) 555-0789\n'),
                len( u'Valerie Richardson\n2906 N. Glenwood Terrace, Atlanta, GA 30310\n(404) 555-0789\nBusiness Operations Specialists'),'POS'), (208, 224, 'FAX'), (186, 202, 'TEL'),(139,181,'add')]),

        (
            u'Dr. Nishantha Panditharathne\nFashion Designer\n+94 11 2686931	+94 11 2686931 ',
            [(len(u'Dr. Nishantha Panditharathne\n'), len(u'Dr. Nishantha Panditharathne\nFashion Designer'), 'POS'),
                (46, 60, 'TEL'), (61, 75, 'TEL')]),
        (
            u'Ms. DD Kariyawasam\nBiochemists\ndamitha@ugc.ac.lk\nTelephone : +94 11 5329500\nExt : 9500\nFax : +94 11 2885358\nEmail : controller[at]immigration.gov.lk\n+94 11 2678731',
            [(
                len(u'Ms. DD Kariyawasam\n'), len(u'Ms. DD Kariyawasam\nBiochemists'), 'POS'), (93, 107, 'FAX'),
                (61, 75, 'TEL')]),
        (
            u'Mr. M. Rajendran\nArt Directors \nBlk 35 Mandalay Road\n 1337 Mandalay Towers\nSINGAPORE 308215\nSINGAPORE',
            [(
                len(u'Mr. M. Rajendran\n'), len(u'Mr. M. Rajendran\nArt Directors'), 'POS'),(32,101,'add')]),
        (
            u'Mayur Hazarika\nGM - Marketing & Sales\nColombo City Centre Patners (Pvt) Ltd.',
            [(
                len(u'Mayur Hazarika\n'), len(u'Mayur Hazarika\nGM - Marketing & Sales'), 'POS')]),
        (
            u'Sr. Francisco Ansó García\nPharmacists \nPaseo de la Castellana, 185, 5ºB\n29001 Madrid\nMadrid',
            [(
                len(u'Sr. Francisco Ansó García\n'), len(u'Sr. Francisco Ansó García\nPharmacists'), 'POS')]),
        (
            u'Monsieur\nGenetic Counselors\nPierre Dupont\nRue Pépinet 10\n1003 Lausanne\nSuisse',
            [(
                len(u'Monsieur\n'), len(u'Monsieur\nGenetic Counselors'), 'POS'),(42,77,'add')]),
        (
            u'Mr Joe Engle\nFinancial Managers\n1612 Dexter Street\nFORT WAYNE IN 46805\nUNITED STATES OF AMERICA',
            [(
                len(u'Mr Joe Engle\n'), len(u'Mr Joe Engle\nFinancial Managers'), 'POS'),(32,95,'add')]),
        (
            u'Dato S.M. Nasrudin\nManaging Director\nCapital Shipping Bhd.\nLot 323, 1st Floor, Bintang Commercial Centre\n29 Jalan Sekilau\n81300 JOHOR BAHRU\nJOHOR\nMALAYSIA',
            [(
                len(u'Dato S.M. Nasrudin\n'), len(u'Dato S.M. Nasrudin\nManaging Director'), 'POS'),(59,154,'add')]),
        (
            u'Nihatha Lathiff\nVice President - Education, IFS Toastmasters Club\nToastmasters International \nWhere Leaders Are Made \nPhone: +94 11 236 4400 Ext 1207\nMobile: +94 77 116 1459\nwww.toastmasters.org\nhttp://www.ifstoastmasters.club/\nsites.google.com/site/ifscmbtoastmasters/',
            [(
                len(u'Nihatha Lathiff\n'), len(u'Nihatha Lathiff\nVice President'), 'POS'), (125, 140, 'TEL'),
                (158, 173, 'MOB')]),
        (
            u'IFS R&D International (Pvt) LtdWebsiteDirections\n4.6\n183 Google reviews\nSoftware company in Colombo, Sri Lanka\nAddress: 501 Galle Rd, Colombo 00600\nHours: Open today · 8AM–5PM\nPhone: 011 2 364400',
            [(183, 195, 'TEL'),(120,147,'add')]),
        (
            u'Coca-Cola Beverages Sri Lanka Ltd\n3.9\n25 Google reviews\nBeverage supplier in Sri Lanka\nAddress: B214 Biyagama Rd\nPhone: 011 2 487700',
            [(120, 135, 'TEL'),(96,112,'add')]),
        (
            u'Arpico Super CenterWebsiteDirections\n4.0\n372 Google reviews\nSupermarket\nAddress: 338, Galle Rd, Wellawatte, Colombo 6\nHours: Open today · 9AM–10PM\nPhone: 011 4 527494',
            [(154, 166, 'TEL'),(81,117,'add')]),
        (
            u'Mohan De Silva\n282, Kaduwela Road, \nBattaramulla.\nSri Lanka\nT: +94 (11) 4714162 /3, 2877300 /1\nF: +94 (11) 4714161',
            [(63, 82, 'TEL'),(84,94,'TEL'),(98,114,'FAX'),(15,59,'add')]),
        (
            u'Susil Rodrigo,\nProject Director,\nColombo City Centre Project.\n295,Madampitiya Road, Colombo 14.\nTel     :    (94) 0112522939  |Fax  :    (94) 0112522942\nWeb    :    www.sankenconstruction.com',
            [(len('Susil Rodrigo,\n'),len('Susil Rodrigo,\nProject Director'), 'POS'), (len('Susil Rodrigo,\nProject Director,\nColombo City Centre Project.\n295,Madampitiya Road, Colombo 14.\nTel     :    '),len('Susil Rodrigo,\nProject Director,\nColombo City Centre Project.\n295,Madampitiya Road, Colombo 14.\nTel     :    (94) 0112522939'), 'TEL'), (len('Susil Rodrigo,\nProject Director,\nColombo City Centre Project.\n295,Madampitiya Road, Colombo 14.\nTel     :    (94) 0112522939  |Fax  :    '),len('Susil Rodrigo,\nProject Director,\nColombo City Centre Project.\n295,Madampitiya Road, Colombo 14.\nTel     :    (94) 0112522939  |Fax  :    (94) 0112522942'), 'FAX'),(62,94,'add')]),
        (
            u'295,Madampitya Road, Colombo 14.\nMobile   :    (94) 0718448016  |Tel   :   (94) 0112522939  |Fax  :   (94) 0112522942',
            [(len('295,Madampitya Road, Colombo 14.\nMobile   :    '),len('295,Madampitya Road, Colombo 14.\nMobile   :    (94) 0718448016'), 'MOB'), (len('295,Madampitya Road, Colombo 14.\nMobile   :    (94) 0718448016  |Tel   :   '),len('295,Madampitya Road, Colombo 14.\nMobile   :    (94) 0718448016  |Tel   :   (94) 0112522939'), 'TEL'), (len('295,Madampitya Road, Colombo 14.\nMobile   :    (94) 0718448016  |Tel   :   (94) 0112522939  |Fax  :   '),len('295,Madampitya Road, Colombo 14.\nMobile   :    (94) 0718448016  |Tel   :   (94) 0112522939  |Fax  :   (94) 0112522942'), 'FAX')]),
        (
            u'Champika Waidyaratne	\nProject Manager - Interiors 	M +94 77 7365733\nJones Lang LaSalle Lanka (Private) Limited	',
            [(22, 37, 'POS'), (53, 67, 'MOB')]),
        (
            u'Faadhil Zainudeen\n+94 (77 5431484)\nowendesignstudio.lk',
            [(18, 34, 'TEL')]),
        (
            u'Colombo City Centre Patners (Pvt) Ltd.\nNo 137, Sir James Pieris Mawatha, Colombo 02.\nMobile: +94-712513213\nEmail : mayur@colombocitycentre.lk',
            [(93, 106, 'MOB')]),
        (
            u'Amanda (Senior Manager)\nMobile No./WhatsApp: 0086 182 5796 0799\nWechat/Skype: kentsales8\nQQ: 22597 13894',
            [(45, 63, 'MOB'),(8,22,'POS'),(42,83,'add')]),
        (
            u'Chamil Jayakody\nHäfele India (Pvt) Limited\nManager – Projects sales :  Häfele Sri Lanka \nMobile +94 77 9804357\nEmail : chamil.jayakody@hafeleindia.com \nhttp://www.hafele.com',
            [(96,110, 'MOB'), (43, 50, 'POS')]),
        (
            u'Chintaka Perera\nExecutive Director\nM: +94 71 864 3564',
            [(38, 53, 'MOB'), (16, 34, 'POS')]),
        (
            u'Chintaka Perera\nExecutive Director\nM: +94 71 864 3564',
            [(38, 53, 'MOB'), (16, 34, 'POS')]),
        (
            u'551, Nawala Road \nRajagiriya 10107\nSri Lanka\nT:            +94 112 877 595\nF:            +94 112 877 595\nE:            chinthaka@vantiles.lk, info@vantiles.lk \nW:          www.vanstrientiles.lk',
            [(59, 74, 'TEL'), (89, 104, 'MOB')]),
        (
            u'Head Office\n113/16 Nawala Road\nNugegoda 10250\nSri Lanka\nT:            +94 11 4513327/8 \nF :           +94 11 2555101 ',
            [(70, 86, 'TEL'), (len('Head Office\n113/16 Nawala Road\nNugegoda 10250\nSri Lanka\nT:            +94 11 4513327/8 \nF :           '),len('Head Office\n113/16 Nawala Road\nNugegoda 10250\nSri Lanka\nT:            +94 11 4513327/8 \nF :           +94 11 2555101'), 'FAX'),(12,55,'add')]),
        (
            u'No.36, Ln. 287, Sec. 5, New Taipei Blvd., Taishan Dist., \nNew Taipei City 243, Taiwan \nTel: +886-2-2296-3999 ext. 2704 ‧ Fax: +886-2-2900-7622\nEmail & Skype: espelin@twkd.com\nCel. & WhatsApp: +886-975-252-905',
            [(92, 108, 'TEL'), (126, 142, 'FAX'),(192,208,'MOB'),(3,85,'add')]),
        (
            u'Randula Jayasuriya.\nAssistant Marketing Manager\nInternational Construction Consortium Pvt Ltd\nNo 70, S De S Jayasinghe Mw,\nKohuwala, Nugegoda,\nSri Lanka.\n\nMob: +94770201487\nTel: +94114400600\nWeb: www.icc-construct.com\nEmail: randula.jayasuriya@icc-construct.com',
            [(20,47,'POS'),(178, 190, 'TEL'), (160, 172, 'MOB'),(97,152,'add')]),
        (
            u'273, DEWALA ROAD, KOSWATTE, BATTARAMULLA, SRI LANKA.\nT: 0094 11 2078118\nM: 0094 772 357561-2\nE: iconcast@yahoo.com\nW: www.iconcast.lk',
            [(56, 71, 'TEL'), (75, 92, 'MOB'),(0,51,'add')]),
        (
            u'T.	(82) 2 3415 4193	|	F.	(82) 2 3415 4165	|	M.	(82) 10 9383 5386\nE.	hglee@kumkangkind.com',
            [(3, 19, 'TEL'), (25, 41, 'FAX'),(47,64,'MOB')]),
        (
            u'14 Robinson Road, #07-01, \nFar East Finance Building, Singapore 048545.\nTel: +65-6383-3661\nFax: +65-6383-2389 ',
            [(77, 90, 'TEL'), (96, 109, 'FAX')]),
        (
            u'DORMA India Private Limited, \n14, Pattulous Road, Chennai - 600002, Tamilnadu, India. \nTel.: +91 44 2858 5097 \nFax: +91 44 4222 7068 \nMB : +94 727 796 796',
            [(93, 109, 'TEL'), (116, 132, 'FAX'),(139,154,'MOB'),(0,70,'add')]),
        (
            u'Abans Engineering (Pvt) Ltd.- MEP Division\n206/8, Lake Drive, Rajagiriya, Sri Lanka\nTel: +94-11-5776969\nMob: +94-77-6761121\nEmail: thusharad@abansgroup.com',
            [(89, 103, 'TEL'), (109, 123, 'MOB'),(43,83,'add')]),
        (
            u'Call: +91 98 11 487450',
            [(len('Call: '), len('Call: +91 98 11 487450'), 'TEL'), ]),
        (
            u'Mobile   :    (94) 0718448016  |Tel   :   (94) 0112522939  |Fax  :   (94) 0112522942\nWeb    :    www.sankenconstruction.com',
            [(len('Mobile   :    '), len('Mobile   :    (94) 0718448016'), 'MOB'), (len('Mobile   :    (94) 0718448016  |Tel   :   '),len('Mobile   :    (94) 0718448016  |Tel   :   (94) 0112522939'), 'TEL'), (len('Mobile   :    (94) 0718448016  |Tel   :   (94) 0112522939  |Fax  :   '),len('Mobile   :    (94) 0718448016  |Tel   :   (94) 0112522939  |Fax  :   (94) 0112522942'), 'FAX')]),
        (
            u'T  +94 11 235 9369\nF  +94 11 230 5776\nD +94 11 235 9348\nM +94 77 356 8371',
            [(3, 18, 'TEL'),(22,37,'FAX'),(40,55,'TEL'),(58,73,'MOB') ]),
        (
            u'Tel: 011 2 67 16 38 Telephone : 014 3481420 Fax: 011 2561713 FAX:011 2343845 Cell :075 4524874 Mobile : 077 3489 305 Phone: 011 7 729729',
            [(5, 19, 'TEL'), (32, 43, 'TEL'), (49, 60, 'FAX'), (65, 76, 'FAX'), (83, 94, 'MOB'), (104, 116, 'MOB'),(124,136,'TEL')]),
        (
            u'No 501, Galle Road, Colombo 06, SRI LANKA\nT: +94 (0) 11 2364 400. F: +94 (0) 11 2364321. M :+94 (0) 7685263254 ',
            [(45, 64, 'TEL'), (69, 87, 'FAX'), (92, 110, 'MOB'),(3,41,'add')]),
        (
            u'Call us imageCALL\n+1 800 520 2653\nSend a Message\nMAIL',
            [(len('Call us imageCALL\n'),len('Call us imageCALL\n+1 800 520 2653'), 'TEL'),]),
        (
            u'John is CEO of ITCookies',
            [(len('John is '), len('John is CEO'), 'POS'), ]),
        (
            u'Pantaloons\nWebsiteDirections\n134 Google reviews\nClothing store in New Delhi, India\nAddress: D-4, South Extension II, Ring Road, New Delhi, Delhi 110049, India\nHours: Open today · 11AM–9PM\nPhone: +91 11 4663 1250',
            [(len('Pantaloons\nWebsiteDirections\n134 Google reviews\nClothing store in New Delhi, India\nAddress: D-4, South Extension II, Ring Road, New Delhi, Delhi 110049, India\nHours: Open today · 11AM–9PM\nPhone: '), len('Pantaloons\nWebsiteDirections\n134 Google reviews\nClothing store in New Delhi, India\nAddress: D-4, South Extension II, Ring Road, New Delhi, Delhi 110049, India\nHours: Open today · 11AM–9PM\nPhone: +91 11 4663 1250'), 'TEL'),(92,158,'add')]),
        (
            u'Address: 155 Gordon Baker Rd #501, North York, ON M2H 3N5, Canada\nHours: Closed now \nPhone: +1 800-387-5757\nProvince: Ontario',
            [(92,107,'TEL'),(9,65,'add')]),
        (
            u'Address: 4/949 St Kilda Rd, Melbourne VIC 3000, Australia\nPhone: +61 1800 335 507',
            [(65,81,'TEL'),(9,57,'add')]),
        (
            u'tel (212) 321-7654 \ncell (917) 654-3210 \nfax (323) 999-8888',
            [(4,18, 'TEL'),(25,39, 'MOB'),(45,59, 'FAX')]),
        (
            u'Mykola Hudkovych\n\nIT Analyst\n\n+380 12 345 6789\n\nMH@post.ua',
            [(18,28, 'POS'), (30, 46, 'TEL')]),
        (
            u'Office: 152-239-9991\nCell: 854-961-9992\nToll-Free: 237-929-9993\nFax: 154-654-9994\nTel: 241-892-1256\nPhone: 210-851-9996\nMobile: 782-629-9997\nF: 529-327-9999',
            [(8, 20, 'TEL'), (27, 39, 'MOB'),(51, 63, 'TEL'), (69, 81, 'FAX'),(87, 99, 'TEL'), (107, 119, 'TEL'),(128, 140, 'MOB'), (144, 156, 'FAX')]),
        (
            u'Our team is happy to help you with any questions or requests!\nUS Sales Team: +1 415 910 4649\nEurope Sales Team: +33 1 42 50 19 43',
            [(77, 92, 'TEL'), (112, 129, 'TEL')]),
        (
            u'Office: 152-239-9991\nCell: 854-961-9992\nToll-Free: 237-929-9993\nFax: 154-654-9994\nTel: 241-892-1256\nPhone: 210-851-9996\nMobile: 782-629-9997\nF: 529-327-9999',
            [(8, 20, 'TEL'), (27, 39, 'MOB'), (51, 63, 'TEL'), (69, 81, 'FAX'), (87, 99, 'TEL'), (107, 119, 'TEL'),
             (128, 140, 'MOB'), (144, 156, 'FAX')]),
        (
            u'B Z L LANKA (PRIVATE) LIMITED\nNo: 122/2,Polhengoda Junction, Kirulapone, Base line road, Colombo - 05, Sri Lanka\nSales Mobile: +94 777730880/ +94 717730880\nSales Email: sramesh@eureka.lk / royalenfield@eureka.lk',
            [(127, 140, 'MOB'), (142, 155, 'MOB'),(30,112,'add')]),
        (
            u'ROYAL ENFIELD SÃO PAULO\nAvenida República do Líbano, 2070 Ibirapuera CEP: 04502-100 São Paulo - SP Brasil\nSales Landline: (+55) 11 5051 7700\nSales Mobile: (+55) 11 954 861 746 / (+55) 11 954 861 718',
            [(122, 140, 'TEL'), (155, 175, 'MOB'),(178, 198, 'MOB'),(24,105,'add')]),
        (
            u'2800 West Big Beaver Road\nSpace U213\nTroy, MI 48084\n(248) 205-5990',
            [(52, 66, 'TEL')]),
        (
            u'Lagoon Cabanas\nAddress Panama\nTelephone 0113818133',
            [(40, 50, 'TEL')]),
        (
            u'Roots - Hyde Park Corner ( Glitz )\nAddress Colombo 02\nTelephone 0113021115',
            [(64, 74, 'TEL')]),
        (
            u'Illiyas Electronics\nAddress No.12 Meeraniya Street Colombo 12\nTelephone 0112322180, 0766960044',
            [(72, 82, 'TEL'),(84,94,'TEL'),(31,61,'add')]),
        (
            u'Akura Higher Educational Institute\nAddress Athurugiriya\nEmailanuraakura@yahoo.com\nTelephone 0779853557, 0115741356',
            [(92, 102, 'TEL'), (104, 114, 'TEL')]),
        (
            u'Toyota Lanka (Private) Limited\nToyota Plaza\nNo.337, Negombo Road, Wattala,\nSri Lanka.General – +94 11 2939000 – 6\nFax         – +94 11 2939005\nEmail     – info@toyota.lk',
            [(95, 109, 'TEL'), (128, 142, 'FAX'),(47,84,'add')]),
        (
            u'Negombo Branch\nNo 375,\nColombo Road,\nNegombo.\nHot Line – +94 31 2220333\nGeneral Line 1 – +94 31 2223810\nLine 2 – +94 31 2223811\nFax Spare Parts – +94 31 2223809\nFax Service –+94 31 2220620',
            [(57, 71, 'TEL'), (89, 103, 'TEL'),(113, 127, 'TEL'), (146, 160, 'FAX'),(174, 188, 'FAX'),(18,44,'add')]),
        (
            u'Forklift Services: 0112939000 Ext. 346\nService Advisor:0777939938 (Kalpa Pathirathna)',
            [(19, 29, 'TEL'), (55, 65, 'TEL')]),
        (
            u'Mr. M. N. Ranasinghe \nTelephone : +94 11 5329500\nExt : 9500\nFax : +94 11 2885358\nEmail : controller[at]immigration.gov.lk',
            [(34, 48, 'TEL'), (66, 80, 'FAX')]),
        (
            u'Kalhari Dissanayake \nPresident | IFS Welfare\nIFS R&D (International) Pvt. Ltd, \n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 011 2364400. Fax +94 011 2364401 \nkalhari.dissanayake@ifsworld.com | www.IFSWORLD.com ',
            [(21, 30, 'POS'), (125, 140, 'TEL'),(146,161,'FAX'),(80,120,'add')]),
        (
            u'Kent Group of Companies\nHead Office\n192, Sir James Peiris Mawatha, \nColombo 2, \nSri Lanka.\nPhone : + 94 112 448844\nFax      : + 94 112 448858',
            [(99, 114, 'TEL'), (126, 141, 'FAX'),(36,89,'add')]),
        (
            u'Mr. Jiffry Farzandh.\nTelephone 011-2698426, ext 232\nEmail	dir.pub@iesl.lk',
            [(31, 42, 'TEL')]),
        (
            u'Buddhi Sathsara Perera\nB.Sc Eng.(Hons), CEng., MIE(SL)\nManager Engineering \nColombo City Center Project\n295,Madampitiya Road, Colombo 14.\nMobile   :    (94) 0712578916 |Tel   :   (94) 0112307191  |Fax  :   (94) 0112307191\nWeb    :    www.sankenconstruction.com',
            [(55, 74, 'POS'),(152,167,'MOB'),(179,194,'TEL'),(206,221,'FAX'),(104,136,'add')]),
        (
            u'Dilon Fernando\nB.Sc Eng.(Hons), AMIE(SL)\nInterior Coordination Engineer\nColombo City Center Project \n295,Madampitiya Road, Colombo 14.\nMobile   :    (94) 0716388895 |Tel   :   (94) 0112522939  |Fax  :   (94) 0112522942\nWeb    :    www.sankenconstruction.com',
            [(41, 71, 'POS'),(149,164,'MOB'),(176,191,'TEL'),(203,218,'FAX'),(101,133,'add')]),
        (
            u'Sagara Akalanka\nB.Sc.Eng(Hons), M.Eng,C.Eng\nDirector \nTEKZOL (PRIVATE) LIMITED,\nNo 164, Stanley Thilakaranthne Mawatha, \nNugegoda 10250\nSri Lanka\nTel : 94 77 7732 946\nE mail :akalanka@tekzol.com ',
            [(44, 52, 'POS'),(152, 166, 'TEL'),(83,145,'add')]),
        (
            u'Thushara Pathirana\nProject Manager (CCC - Hotel Scope)\nB.Sc Eng (Hon’s), AMIESL, MASHRAE\nAbans Engineering (Pvt) Ltd.- MEP Division\n206/8, Lake Drive, Rajagiriya, Sri Lanka',
            [(19, 34, 'POS'),(132,172,'add')]),
        (
            u'Hee Gyo, Lee (Bob)		 \nOverseas Business Div.\nT.	(82) 2 3415 4193	|	F.	(82) 2 3415 4165	|	M.	(82) 10 9383 5386\nE.	hglee@kumkangkind.com\nW.	www.kumkangkind.com',
            [(48, 64, 'TEL'), (70, 86, 'FAX'),(92, 109, 'MOB')]),
        (
            u'Level 10 & 11\nWest Tower, World Trade Center,\nEchelon Square\nColombo 01, Sri Lanka\nBusiness Inquiries: +94 (0) 75 955 7000\nEmail: 555@airtel.lk\nFax: +94 (0) 112 448 933/4\nCustomer Care: 1755 from your Airtel number \nor +94 (0) 755 555 555',
            [(103, 122, 'TEL'),(149,170,'FAX'),(219,238,'MOB'),(0,82,'add')]),
        (
            u'Information and Communication Technology Agency of Sri Lanka\n160/24,Kirimandala Mawatha,Colombo 5.\n00500\nSri Lanka\nTelephone: 	+94 11 2 369099 to 100, ext 355\nFax: 	+94 11 2 368387',
            [(127, 142, 'TEL'), (165, 180, 'FAX'),(61,114,'add')]),
        (
            u'Hugo Halimi\nBusiness Development Specialist | Evercontact\ne: hugo@onemore.company | cell: 415 910-4649\n25 West 39th Street, 14th Floor\nNew York, NY, 10018',
            [(12, 43, 'POS'), (90, 102, 'MOB'),(103,154,'add')]),
        (
            u'Phone:\n+94 11 2486000\n+94 11 4486000\nSwift:\nCCEYLKLX\nSkype:\nCommercial Bank Call Center\nEmail:\ninfo@combank.net\nFax:\n+94 11 2449889\nTelebanking:\n+94 11 2336633',
            [(7, 21, 'TEL'), (22, 36, 'TEL'),(117, 131, 'FAX'), (145,159, 'TEL')]),
        (
            u'Sri Lanka Customs,\nNo.40, Main Street, Colombo 11, Sri Lanka.\nTele: +94 11 2470945-8\nTele: +94 11 2445147\nFax: +94 11 2446364',
            [(68, 84, 'TEL'), (91, 105, 'TEL'), (111, 125, 'FAX'),(22,60,'add')]),
        (
            u'General Inquiries : 011 2661995 (Working Hours)\nE-Banking : 011 2462462 (24 hour contact centre)\nCard Centre : 011 2462462 (24 hour contact centre)\nFax : 011 2662759\nE-mail : feedback@hnb.lk',
            [(20, 31, 'TEL'), (60, 71, 'TEL'), (111, 122, 'TEL'), (154, 165, 'FAX')]),
        (
            u'E-mail: \ncypher@mfa.gov.lk\nTelephone: \n0094 (0) 11 2325371 / 2325372 / 2325373 / 2325375\nFax: \n0094 (0) 11 2446091 / 2333450 / 2430220',
            [(39, 58, 'TEL'), (61, 68, 'TEL'), (71, 78, 'TEL'), (81, 88, 'TEL'),(95, 114, 'FAX'), (117, 124, 'FAX'), (127, 134, 'FAX')]),
        (
            u'Chintaka Perera\nHR Manager\nM: 071 864 3564\nTel 021 5684927\nFax 011 584 1328',
            [(16, 26, 'POS'), (30, 42, 'MOB'),(47,58,'TEL'),(63,75,'FAX')]),
        (
            u'Sheldon Cooper\nSenior Theoretical Physicist\nCalifornia Institute of Technology\n1200 E California Blvd,\nPasadena, CA 91125, United States\nsheldon@example.com\nphone-(626) 555-9157\nfax-(626) 555-4717',
            [(15, 43, 'POS'), (163, 177, 'TEL'), (182, 196, 'FAX'),(79,136,'add')]),
        (
            u'Nirmani Wijesinghe\nHR Executive\nIFS R&D International (Pvt) Ltd\n501, Galle Road, Colombo 06,   SRI LANKA\nTel +94 011 2364400. Fax +94 011 2364401 ',
            [(19, 31, 'POS'), (109, 124, 'TEL'), (130, 145, 'FAX'),(64,104,'add')]),
        (
            u'SoftLogic FinanceWebsiteDirections\n3.0\n5 Google reviews\nFinancial consultant in Colombo, Sri Lanka\nAddress: 13 De Fonseka Pl, Colombo 00500\nHours: Open today · 9AM–5PM\nPhone: 011 2 359700',
            [(175, 187, 'TEL'),(108,139,'add')]),
        (
            u'Bernie Reeder\nContent Engagement Manager,Yesware\n855-937-9273',
            [(14, 40, 'POS'),(49,61,'TEL')]),
        (
            u'Hours: Open today · 5AM–11PM\nMenu: starbucks.com\nFax : +1 813-928-7580',
            [(55, 70, 'FAX'),]),
        (
            u'Hours: Open today · 5AM–11PM\nMenu: starbucks.com\nMob : +1 813-928-7580',
            [(55, 70, 'MOB'), ]),
        (
            u'Hours: Open today · 5AM–11PM\nMenu: starbucks.com\nMobile : +1 813-928-7580',
            [(58, 73, 'MOB'), ]),
        (
            u'Hours: Open today · 5AM–11PM\nMenu: starbucks.com\nCell : +1 813-928-7579',
            [(56, 71, 'MOB'), ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nM: +44 20 3510 0444',
            [(95, 111, 'MOB'),(9,62,'add')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nM +44 20 3510 0444',
            [(94, 110, 'MOB'), (9,62,'add')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nMOB +44 20 3510 0444',
            [(96, 112, 'MOB'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nMOBILE +44 20 3510 0444',
            [(99, 115, 'MOB'),  (9,62,'add')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nPHONE +44 20 3510 0444',
            [(98, 114, 'TEL'),  (9,62,'add')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nFax +44 20 3510 0444',
            [(96, 112, 'FAX'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nmobile: +44 20 3510 0444',
            [(100, 116, 'MOB'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nmob: +44 20 3510 0444',
            [(97, 113, 'MOB'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nm: +44 20 3510 0444',
            [(95, 111, 'MOB'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nm: + 1 918 - 331 - 9436',
            [(95, 115, 'MOB'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nMob : + 1 918 - 331 - 9436',
            [(98, 118, 'MOB'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nMob : 011 236 4585',
            [(98, 110, 'MOB'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nT : 011 236 4585',
            [(96, 108, 'MOB'),  (9,62,'add')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nTEL : 011 236 4585 ',
            [(98, 110, 'TEL'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nTELE : 011 236 2514 Tel 1-800-500-9862',
            [(99, 111, 'TEL'),(116,130,'TEL'), (9,62,'add') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nTele : 081 239 2894 phone:+61 8 82326262 P +1 555 123-4567 P: (555) 123-4567',
            [(99, 111, 'TEL'),(118,132,'TEL'),(135,150,'TEL'),(154,168,'TEL') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nF : 081 239 2894',
            [(96, 108, 'FAX'), ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nt : 081 239 2894 T-+1 202-635-0088 Phone - +33 3 28 09 02 20 f: +94 56 852 2596',
            [(96, 108, 'TEL'),(111,126,'TEL'),(135,152,'TEL'),(156,171,'FAX')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nTel: +94 081 239 2894 Telephone Number-+1 202-635-0088 ',
            [(97, 113, 'TEL'),(131,146,'TEL')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nT: +94 081 239 2894, OFFICE: (323) 467-1175 +61 1300 749 924 ',
            [(95, 111, 'TEL'),(121,135,'TEL'),(136,152,'TEL')]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\nTel: +94 081 239 2894 O +44 85 121 4523 TEL +49 30 26930741',
            [(97, 113, 'TEL'),(116,131,'TEL'),(136,151,'TEL') ]),
        (
            u'Address: Edinburgh Airport, Edinburgh EH12 9DN, United Kingdom\nHours: Open today · 4AM–11PM\n+94 081 239 2894 Fax Number:+94 064 258 14 13 Line:+33 1 48 55 48 57',
            [(92, 108, 'TEL'),(120,137,'FAX'),(143,160,'TEL') ]),
        (
            u'Address: Hauptmarkt 1, 90403 Nürnberg, Germany\nHours: Open today · 8AM–8:30PM\nTel- +49 911 23555910 Office - +44 50 7541 8523 CELL: 415-861-1313',
            [(83, 99, 'TEL'),(109,125,'TEL'),(132,144,'MOB'),(9,46,'add') ]),
        (
            u'Address: Hauptmarkt 1, 90403 Nürnberg, Germany\nHours: Open today · 8AM–8:30PM\nMob- +49 911 23555910 Fax +94 044 8542168 Fax:+94 031 8526932. F  +44 1-2222 8888 F - 001 (212) 999 8888',
            [(83, 99, 'MOB'),(104,119,'FAX'),(124,139,'FAX'),(144,159,'FAX'),(164,182,'FAX'),(9,46,'add')]),
        (
            u'Address: Hauptmarkt 1, 90403 Nurnberg, Germany\nHours: Open today · 8AM–8:30PM\nFax- +49 911 23555910 Tel +41 848 77 66 55 F +93 74 785 15 26 \n(212) 555-0989\n212-555-0989\n212.555.0989',
            [(83, 99, 'FAX'),(104,120,'TEL'),(123,139,'FAX'),(141,155,'TEL'),(156,168,'TEL'),(169,181,'TEL'),(9,46,'add')]),
        (
            u'KFC George St\n485 George St · +61 2 9283 3915\nOpen until 2:00 AM.M 281-554-9635 P 254-956-8430 F 857-634-8852 Cell: (917) 555-1987 Mobile: 917-555-1987 FAX +1 212 999 8888 ',
            [(30, 45, 'TEL'),(67,79,'MOB'),(82,94,'TEL'),(97,109,'FAX'),(116,130,'MOB'),(139,151,'MOB'),(156,171,'FAX')]),
        (
            u'Address: Bishan Community Club 51 Bishan Street 13 #01-02, Singapore 579799\nHours: Open today · 7:30AM–11PM\nPhone: +65 6910 1239 P: 125-894-5216 F: 851-962-5843 M: 895-452-9648',
            [(115, 128, 'TEL'), (132, 144, 'TEL'), (148, 160, 'FAX'), (164, 176, 'MOB'),(9,75,'add')]),
        (
            u'200 University Avenue West, Waterloo, ON, N2L 3G1\nP 519-888-4567, ext. 77777\nC 226-845-5412 C : 529-874-6630 fax +1 (212) 222 8888.f  1-212-222 8888 f - 0161 999 8888 Tel - 041 444-5555 F +44 41 444-5555 f +44 41 444-5555 fax +44 41 444-5555 T +44 41 444-5555',
            [(52, 64, 'TEL'), (79, 91, 'MOB'),(96,108,'MOB'),(113,130,'FAX'),(134,148,'FAX'),(153,166,'FAX'),(173,185,'TEL'),(188,203,'FAX'),(206,221,'FAX'),(226,241,'FAX'),(244,259,'TEL'),(0,49,'add')]),
        (
            u'Address: 439 U.S. 81, Concordia, KS 66901, USA\nPhone: +1 785-243-1071',
            [(54, 69, 'TEL'),(9,46,'add') ]),
        (
            u'Voice: 415-961-4111\nFAX: 415-967-9231 Mob 077 58236965  Mobile 077 8340 306  FAX 011 5236987 fax +91 061 458 5845 Tel +1 800 520 2653 Mob +1 800 520 2653 ',
            [(7, 19, 'TEL'),(25,37,'FAX'),(42,54,'MOB'),(63,75,'MOB'),(81,93,'FAX'),(118,133,'TEL'),(138,153,'MOB')]),
        (
            u'Registered office: Teknikringen 5, Box 1545, SE-581 15 Linkoping\nm +94 075 56 47945 \nmob +94 075 56 47945\n MOB +94 075 56 47945\n Mob +94 075 56 47945 \nfax +94 075 56 47945 \n Fax +94 075 56 47945',
            [(67, 83, 'MOB'), (89, 105, 'MOB'), (111, 127, 'MOB'), (133, 149, 'MOB'), (155, 171, 'FAX'),(178,194,'FAX'),(19,64,'add')]),
        (
            u'Address: 3801 E Frank Phillips Blvd, Bartlesville, OK 74006, USA\nHours: Open today · 5:30AM–10PM\nMenu: starbucks.com\nPhone: +1 918-331-9436 Fax: +1 918-331-9436 Mob: +1 918-331-9436 Mobile:  +1 918-331-9436 Tel: +1 918-331-9436 T: +1 918-331-9436 F: +1 918-331-9436 M: +1 918-331-9436',
            [(124, 139, 'TEL'), (145, 160, 'FAX'), (166, 181, 'MOB'), (191, 206, 'MOB'), (81, 93, 'FAX'),(212,227,'TEL'),(231,246,'TEL'),(250,265,'FAX'),(269,284,'MOB'),(9,64,'add')]),
        (
            u'Chamila Wijesinghe\nBUSINESS SYSTEMS ANALYST\nManufacturing | R&D\n501 Galle Road, Colombo 6, SRI LANKA\nTel +94 11 236 44 00. Fax +94 11 236 44 01. Mobile +94 0718669631\nchamila.wijesinghe@ifsworld.com | www.IFSWORLD.com \nIFS World Operations AB is a limited liability company registered in Sweden. \nCorporate identity number: 556040-6042. \nRegistered office: Teknikringen 5, Box 1545, SE-581 15 Linköping.\n??Please consider the environment before printing my email ',
            [(19, 43, 'POS'), (105, 121, 'TEL'), (127, 143, 'FAX'), (152, 166, 'MOB'),(64,100,'add')]),
        ##########################
        (
            u'219 Google reviews\nFast FoodRestaurant\nAddress: York St, Colombo',
            [(48, 64, 'add')]),
        (
            u'Fast-food chain known for its buckets of fried chicken, plus wings & sides.\nAddress: 2807 W Irving Park Rd, Chicago, IL 60618, USA\nHours: Closing soon · 10AM–11PM\nMenu: kfc.com',
            [(85, 130, 'add')]),
        (
            u'Address: Teknikringen 5, 583 30 Linköping, Sweden\nPhone: +46 13 460 40 00',
            [(9, 49, 'add')]),
        (
            u'Address: 820/818 Old Princes Hwy, Sydney NSW 2232, Australia\nHours: Open today · 11AM–10PM\nMenu: pizzahut.com.au',
            [(9, 60, 'add')]),
        (
            u'Located in: United Square\nAddress: 101 Thomson Rd, Singapore 307591\nHours: Open today · Open 24 hours\nPhone: +65 6910 1185',
            [(35, 67, 'add')]),
        (
            u'Fast-food chain known for its buckets of fried chicken, plus wings & sides.\nLocated in: Westfield Knox\nAddress: Knox City Centre, 16 Melbourne St, Wantirna South VIC 3152, Australia\nHours: Open today · 10:30AM–10PM\nPhone: +61 3 9801 9192',
            [(112, 181, 'add')]),
        (
            u'Located in: Alto Palermo Shopping\nAddress: Av. Santa Fe 3253, C1425 CABA, Argentina\nHours: Open today · 10AM–11PM',
            [(43, 83, 'add')]),
        (
            u'807 Google reviews\nFast Food Restaurant\nAddress: Carlos Pellegrini 435, Buenos Aires, Argentina\nHours: Open today · 8AM–11PM',
            [(49, 95, 'add')]),
        (
            u'Fast-food chain known for its buckets of fried chicken, plus wings & sides.\nAddress: 66 Rue du Commerce, 51350 Cormontreuil, France\nHours: Open today · 11AM–11PM\nPhone: +33 3 26 82 93 49',
            [(85, 131, 'add')]),
        (
            u'Seattle-based coffeehouse chain known for its signature roasts, light bites and WiFi availability.\nAddress: 55 Esk St, Invercargill, 9810, New Zealand\nHours: Closed now \nPhone: +64 3-214 0117',
            [(108, 150, 'add')]),
        (
            u'Triesterstrasse 217\n10117 Berlin\nDeutschland\nwww.coca-cola.com',
            [(0, 44, 'add')]),
        (
            u'Fast-food chain known for its buckets of fried chicken, plus wings & sides.\nAddress: Mira Avenue, 45, Krasnoyarsk, Krasnoyarskiy kray, Russia, 660049\nHours: Open today · 8AM–3AM\nPhone: +7 800 555-83-33',
            [(85, 149, 'add')]),
        (
            u'Address: Unnamed Road, 2080, Woodlands, Sandton, South Africa\nPhone: +27 11 563 4600',
            [(9, 61, 'add')]),
        (
            u'Address: Avenida Francisco de Miranda, Caracas, Miranda, Venezuela\nHours: Open today · 9AM–5PM\nPhone: +58 500-4683700',
            [(9, 66, 'add')]),
        (
            u'Address: Mariano Sánchez Fontecilla 310, Las Condes, Región Metropolitana, Chile\nPhone: +56 2 2335 7756',
            [(9, 80, 'add')]),
        (
            u'Address:B2, Dalang Industry Zone, Xiahua 1st Road, Baiyun District\nZip:510000\nCountry/Region:China (Mainland)\nProvince/State: Guangdong',
            [(8, 66, 'add')]),
        (
            u'Mobile Phone:\nFax:\nView Details\nAddress:\n Room 4B-31, Block B, Overseas Decoration Building, Huaqiang North, Futain District.\nZip:\n518000\nCountry/Region:\nChina (Mainland)\nProvince/State:\nGuangdong',
            [(42, 124, 'add')]),
        (
            u'Address:	608, Unit 2,Building 6,Beiqing Road No 1,Changping District, Beijing , 102206, China\nZip:	102206\nCountry/Region:	China (Mainland)',
            [(9, 93, 'add')]),
        (
            u'**\nAddress:\n2201-2203,China South Development Center, China South City, Pinghu, LongGang District\nZip:\n518000\nCountry/Region:\nChina (Mainland)',
            [(12, 97, 'add')]),
        (
            u'Address:	Area H, 2/F, No. 59, Longjing 2nd Rd., Baocheng 3rd Dist., Baoan Dist.\nZip:	518101\nCountry/Region:	China (Mainland)\nProvince/State:	Guangdong\nCity:	Shenzhen',
            [(9, 78, 'add')]),
        (
            u'Telephone:\n**\nAddress:\nB601,ZhongShun Building,Baohua Road,LongHua District\nCountry/Region:\nChina (Mainland)\nProvince/State:',
            [(23, 75, 'add')]),
        (
            u'Address:\nRoom 401 ,Floor 4,Building 2,Jinfang Hua Industrial Park,Yang Mei,Bantian Street,Long Gang District,\nZip:\n518021\nCountry/Region:\nChina (Mainland)\nProvince/State:\nGuangdong',
            [(9, 108, 'add')]),
    ###########################
        (
            u'Address:\n3rd Floor,Elevator A,Building D3,Hengda Industrial Garden,No.3 Bigui Road,luo pu Street,Panyu District, Guangzhou City\nZip:\n510045\nCountry/Region:\nChina (Mainland)\nProvince/State:\nGuangdong',
            [(9, 127, 'add')]),
        (
            u'Address:\n33C Block C ShenNan Road2070, FuTian District\nZip:\n518031\nCountry/Region:\nChina (Mainland)\nProvince/State:\nGuangdong',
            [(9, 54, 'add')]),
        (
            u'Address:\n7F A1 Bldg, HuaFeng Industry Area, Gushu Town, Baoan District\nZip:\nCountry/Region:\nChina (Mainland)\nProvince/State:\nGuangdong',
            [(9, 70, 'add')]),
        (
            u'Address:\nNo.2,Quanjian Road,Douzhangzhuang Town,Wuqing District\nZip:\nCountry/Region:\nChina (Mainland)\nProvince/State:\nTianjin',
            [(9, 63, 'add')]),
        (
            u'Address:\nRm. 610, 6/F, Art Apia Building, South Zhongkang Road, Meilin, Futian District\nZip:\n518049\nCountry/Region:\nChina (Mainland)\nProvince/State:\nGuangdong',
            [(9, 87, 'add')]),
        (
            u'Address:\n7F A1 Bldg, HuaFeng Industry Area, Gushu Town, Baoan District\nZip:\nCountry/Region:\nChina (Mainland)\nProvince/State:\nGuangdong',
            [(9, 70, 'add')]),
        (
            u'Microsoft Sri LankaWebsiteDirections\n92 Google reviews\nCondominium complex in Colombo, Sri Lanka\nAddress: 11th Floor, DHPL Building, No. 42, Navam Mawatha, Colombo',
            [(106,163, 'add')]),
        (
            u'Elon Musk\nCEO of SpaceX',
            [(10, 13, 'add')]),
        (
            u'Luis Escala\nPiedras 623\nPiso 2, depto 4\nC1070AAM, Capital Federal',
            [(12, 65, 'add')]),
        (
            u'Ms H Williams\nFinance and Accounting\nAustralia Post\n219–241 Cleveland St\nSTRAWBERRY HILLS  NSW  1427',
            [(52, 100, 'add')]),
        (
            u'Mr J. ODonnell\nLighthouse Promotions\nPO Box 215\nSPRINGVALE  VIC  3171',
            [(37, 69, 'add')]),
        (
            u'Carlos Rossi\nAvenida João Jorge, 112, ap. 31\nVila Industrial\nCampinas - SP\n13035-680',
            [(13, 84, 'add')]),
        (
            u'Sr. Rodrigo Domínguez\nAv. Bellavista N° 185\nDep. 609\n8420507\nRecoleta\nSantiago',
            [(22, 78, 'add')]),
        (
            u'Ranvir Singh\n5, Mahatma Gandhi Road\nBUDHAGAON\nDistrict Sangli\n471594\nMaharashtra',
            [(13, 80, 'add')]),
        (
            u'Mr A Smith\n3a High Street\nHedge End\nSOUTHAMPTON\nSO31 4NG',
            [(11, 56, 'add')]),
        (
            u'Jeremy Martinson, Jr.\n455 Larkspur Dr.\nBaviera, CA 92908',
            [(22, 56, 'add')]),
        (
            u'Mr. Wang\n2F., No.2, Shifu Rd.\nXinyi Dist., Taipei City 11060\nTaiwan',
            [(9, 67, 'add')]),
        (
            u'Mohammed Ali Al-Ahmed\n8228 Imam Ali Road – Alsalam Neighbourhood\nRiyadh 12345-6789\nKingdom of Saudi Arabia',
            [(22, 106, 'add')]),
        (
            u'Thomas van der Landen\nAddress:  Boschdijk 1092 5631 AV EINDHOVEN Nederland',
            [(32, 74, 'add')]),
        (
            u'Kari Normann\nAddress :  Storgata 81A 6415 Molde Norway',
            [(24, 54, 'add')]),
        (
            u'Coca Cola\nAddress: Stralauer Allee 4 10245Berlin Germany',
            [(19, 56, 'add')]),
        (
            u'Stralauer Allee 4\n10245Berlin\nGermany\nwww.cceag.de/',
            [(0, 37, 'add')]),
        (
            u'Stationsstrasse 33\n8306Brüttisellen\nSwitzerland\nPhone: 0448359111\nwww.coca-colahellenic.ch',
            [(0, 47, 'add'),(55,65,'TEL')]),
        (
            u'Am Campeon 10-12\n85579Neubiberg\nGermany\nPhone: +49 (0) 89-9988 53-0\nFax: +49 (0) 89-90439-48',
            [(0, 39, 'add'), (47, 67, 'TEL'),(73,92,'FAX')]),
        (
            u'M. Jen Durand\nAddress:150 Rue Nepeau App5 OTTAWA ON K1P 2P6 CANADA',
            [(22, 66, 'add'),]),
        ######################################
        (
            u'Fax:\nView Details\nAddress: huku town\nZip: 321300\nCountry/Region:China (Mainland)\nProvince/State: Zhejiang ',
            [(27, 36, 'add')]),
        (
            u'Address:Avenida Francisco de Miranda, Caracas, Miranda, Venezuela\nHours: Open today · 9AM–5PM\nPhone: +58 500-4683700',
            [(8, 65, 'add')]),
        (
            u'Address:Mariano Sánchez Fontecilla 310, Las Condes, Región Metropolitana, Chile\nPhone: +56 2 2335 7756',
            [(8, 79, 'add')]),
        (
            u'Address : Avenida Francisco de Miranda, Caracas, Miranda, Venezuela\nHours: Open today · 9AM–5PM\nPhone: +58 500-4683700',
            [(10, 67, 'add')]),
        (
            u'Address : Mariano Sánchez Fontecilla 310, Las Condes, Región Metropolitana, Chile\nPhone: +56 2 2335 7756',
            [(10, 81, 'add')]),
        (
            u'Address :  Avenida Francisco de Miranda, Caracas, Miranda, Venezuela\nHours: Open today · 9AM–5PM\nPhone: +58 500-4683700',
            [(11, 68, 'add')]),
        (
            u'Address :  Mariano Sánchez Fontecilla 310, Las Condes, Región Metropolitana, Chile\nPhone: +56 2 2335 7756',
            [(11, 82, 'add')]),
        (
            u'Microsoft Sri LankaWebsiteDirections\n4.5\n92 Google reviews\nCondominium complex in Colombo, Sri Lanka\nAddress: 11th Floor, DHPL Building, No. 42, Navam Mawatha, Colombo\nPhone: 0114 765 500\nSuggest an edit · Own this business?',
            [(110, 167, 'add'), (175, 187, 'TEL')]),

    ]
    ner = train_ner(nlp, train_data, ['POS','FAX','TEL','MOB','add'])
    f = open('D:\PYTHON\Input\Source.txt')
    string = f.read()
    string2 = unicode(string, "utf-8")
    doc = nlp.make_doc(string2)
    doc1 = unicode(doc)
    nlp.tagger(doc)
    ner(doc)
    tagged_sent = []
    def get_continuous_chunks(tagged_sent):
        continuous_chunk = []
        current_chunk = []

        for text, tag, num in tagged_sent:
            if num != 2 and (tag == 'POS' or tag == 'FAX' or tag == 'TEL' or tag == 'MOB' or tag == 'add'):
                current_chunk.append((text, tag, num))
            else:
                if current_chunk:  # if the current chunk is not empty
                    continuous_chunk.append(current_chunk)
                    current_chunk = []
        # Flush the final current_chunk into the continuous_chunk, if any.
        if current_chunk:
            continuous_chunk.append(current_chunk)
        return continuous_chunk

    for word in doc:
        tagged_sent.append((word.text, word.ent_type_, word.ent_iob))
        # print(word.text, word.ent_type_, word.ent_iob)
    # print(tagged_sent)

    named_entities = get_continuous_chunks(tagged_sent)
    named_entities_str = [" ".join([tag for tag, text, num in ne]) for ne in named_entities]
    named_entities_str_tag = [(" ".join([tag for tag, text, num in ne]), ne[0][1]) for ne in named_entities]
    # print(named_entities)
    # print
    # print(named_entities_str)
    # print
    # print(named_entities_str_tag)
    for x,y in named_entities_str_tag:
        if y == 'FAX':
            print ('FAX :',x)
        elif y== 'TEL':
            print ('Telephone :',x)
        elif y == 'POS':
            print('Position :',x)
        elif y == 'MOB':
            print('Mobile :',x)
        elif y == 'add':
            print('Address :',x)

    if model_dir is not None:
        save_model(ner, model_dir)

if __name__ == '__main__':
    main('ner')
