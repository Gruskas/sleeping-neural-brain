from __future__ import annotations

import json
import re
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .web_learner import is_web_boilerplate

_UA = "SleepingNeuralBrain/2.0 (Educational Fact Harvester; https://localhost)"

# Over 105 verified, crisp Polish encyclopedic entries across 10 domains.
# Each entry provides a concise canonical fact, natural Q&A, and a multi-turn follow-up.
CURATED_FACT_ENCYCLOPEDIA: list[dict[str, str]] = [
    # ── 1. Fizyka (11) ───────────────────────────────────────────────────────
    {
        "domain": "fizyka",
        "title": "Prędkość światła w próżni",
        "fact": "Prędkość światła w próżni wynosi dokładnie 299 792 458 metrów na sekundę i stanowi maksymalną prędkość przekazywania informacji we Wszechświecie.",
        "q": "Ile wynosi prędkość światła w próżni?",
        "a": "Prędkość światła w próżni wynosi dokładnie 299 792 458 metrów na sekundę (ok. 300 tys. km/s) i jest górną granicą szybkości ruchu informacji oraz materii.",
        "followup_q": "Czy cokolwiek z masą spoczynkową może ją osiągnąć?",
        "followup_a": "Nie — według szczególnej teorii względności rozpędzenie ciała o niezerowej masie spoczynkowej do prędkości światła wymagałoby nieskończonej energii.",
    },
    {
        "domain": "fizyka",
        "title": "Zasady dynamiki Newtona",
        "fact": "Trzy zasady dynamiki Newtona opisują bezwładność ciał, związek siły z przyspieszeniem (F = m * a) oraz równość sił akcji i reakcji.",
        "q": "Co opisuje druga zasada dynamiki Newtona?",
        "a": "Druga zasada dynamiki Newtona mówi, że przyspieszenie ciała jest wprost proporcjonalne do wypadkowej siły i odwrotnie proporcjonalne do jego masy: F = m * a.",
        "followup_q": "A czego dotyczy trzecia zasada?",
        "followup_a": "Trzecia zasada stwierdza, że każda akcja wywołuje reakcję o tej samej wartości i kierunku, lecz przeciwnym zwrocie, przyłożoną do drugiego ciała.",
    },
    {
        "domain": "fizyka",
        "title": "Termodynamika i entropia",
        "fact": "Pierwsza zasada termodynamiki wyraża zachowanie energii, a druga mówi, że entropia układu izolowanego nie maleje podczas procesów samorzutnych.",
        "q": "Co oznacza pojęcie entropii w termodynamice?",
        "a": "Entropia jest miarą nieuporządkowania układu oraz rozproszenia energii — w układzie izolowanym w procesach naturalnych całkowita entropia zawsze rośnie lub pozostaje stała.",
        "followup_q": "Dlaczego ciepło nie płynie samo z zimnego ciała do ciepłego?",
        "followup_a": "Ponieważ taki samorzutny przepływ zmniejszałby całkowitą entropię układu, co jest niezgodne z drugą zasadą termodynamiki.",
    },
    {
        "domain": "fizyka",
        "title": "Prawo Ohma i prąd elektryczny",
        "fact": "Prąd elektryczny to uporządkowany ruch ładunków, a prawo Ohma wiąże napięcie U, natężenie I i opór R zależnością U = I * R.",
        "q": "Jak działa prawo Ohma w obwodach elektrycznych?",
        "a": "Prawo Ohma mówi, że w przewodniku o stałej temperaturze natężenie prądu jest proporcjonalne do przyłożonego napięcia i odwrotnie proporcjonalne do oporu: I = U / R.",
        "followup_q": "W jakich jednostkach mierzy się napięcie, natężenie i opór?",
        "followup_a": "Napięcie mierzy się w woltach (V), natężenie prądu w amperach (A), a opór elektryczny w omach (Ω).",
    },
    {
        "domain": "fizyka",
        "title": "Grawitacja i zakrzywienie czasoprzestrzeni",
        "fact": "W ogólnej teorii względności Einsteina grawitacja nie jest zwykłą siłą na odległość, lecz skutkiem zakrzywienia czasoprzestrzeni przez masę i energię.",
        "q": "Jak Einstein wyjaśnił zjawisko grawitacji?",
        "a": "Albert Einstein opisał grawitację jako zakrzywienie geometrii czterowymiarowej czasoprzestrzeni wywołane obecnością masy oraz energii, wzdłuż którego poruszają się ciała i światło.",
        "followup_q": "Co dzieje się wtedy z biegiem czasu?",
        "followup_a": "W silniejszym polu grawitacyjnym czas płynie wolniej niż w przestrzeni oddalonej od dużych mas — zjawisko to nazywamy dylatacją grawitacyjną czasu.",
    },
    {
        "domain": "fizyka",
        "title": "Rozszczepienie światła i widmo elektromagnetyczne",
        "fact": "Białe światło słoneczne składa się z fal o różnych długościach, które po przejściu przez pryzmat ulegają różnemu załamaniu i tworzą tęczowe widmo.",
        "q": "Dlaczego pryzmat rozszczepia białe światło?",
        "a": "Współczynnik załamania szkła zależy od długości fali świetlnej, dlatego fiolet skręca najmocniej, a czerwień najsłabiej, rozdzielając wiązkę na barwy składowe.",
        "followup_q": "Co oprócz światła widzialnego należy do fal elektromagnetycznych?",
        "followup_a": "Fale radiowe, mikrofale, podczerwień, ultrafiolet, promieniowanie rentgenowskie oraz najbardziej energetyczne promieniowanie gamma.",
    },
    {
        "domain": "fizyka",
        "title": "Dźwięk i fala akustyczna",
        "fact": "Dźwięk jest falą mechaniczną rozchodzącą się w ośrodku sprężystym; w suchym powietrzu przy 20°C jego prędkość wynosi około 343 metry na sekundę.",
        "q": "Dlaczego w próżni kosmicznej nie słychać dźwięku?",
        "a": "Dźwięk jest falą mechaniczną wymagającą cząsteczek ośrodka (gazu, cieczy lub ciała stałego) do przenoszenia drgań, a w próżni brakuje materii zdolnej je przewodzić.",
        "followup_q": "Od czego zależy wysokość słyszanego dźwięku?",
        "followup_a": "Wysokość dźwięku zależy od częstotliwości drgań mierzonej w hercach (Hz) — wyższa częstotliwość daje wyższy ton.",
    },
    {
        "domain": "fizyka",
        "title": "Zjawisko fotoelektryczne i fotony",
        "fact": "Światło wykazuje dualizm korpuskularno-falowy: składa się z kwantów energii zwanych fotonami, których energia rośnie wraz z częstotliwością fali.",
        "q": "Co udowodnił Einstein wyjaśniając efekt fotoelektryczny?",
        "a": "Wykazał, że światło przenosi energię w dyskretnych porcjach zwanych fotonami, a wybicie elektronu z metalu zależy od częstotliwości pojedynczego fotonu, nie zaś samego natężenia.",
        "followup_q": "Gdzie stosuje się to zjawisko dzisiaj?",
        "followup_a": "W ogniwach fotowoltaicznych (panelach słonecznych), fotodiodach oraz matrycach aparatów cyfrowych.",
    },
    {
        "domain": "fizyka",
        "title": "Nadprzewodnictwo elektryczne",
        "fact": "Nadprzewodnictwo to stan materiału poniżej temperatury krytycznej, w którym opór elektryczny spada dokładnie do zera i zanika wewnętrzne pole magnetyczne.",
        "q": "Czym wyróżnia się materiał w stanie nadprzewodnictwa?",
        "a": "Przewodzi prąd całkowicie bez oporu i strat ciepła, a także wypycha ze swojego wnętrza linie pola magnetycznego, co nazywamy efektem Meissnera.",
        "followup_q": "Do czego używa się nadprzewodników?",
        "followup_a": "Do budowy potężnych elektromagnesów w skanerach rezonansu magnetycznego (MRI), akceleratorach cząstek takich jak LHC oraz pociągach lewitujących maglev.",
    },
    {
        "domain": "fizyka",
        "title": "Ciśnienie atmosferyczne i prawo Pascala",
        "fact": "Standardowe ciśnienie atmosferyczne na poziomie morza wynosi 1013,25 hPa, a ciśnienie wywierane na zamknięty płyn rozchodzi się równomiernie we wszystkich kierunkach.",
        "q": "Jak działa prasa hydrauliczna w oparciu o prawo Pascala?",
        "a": "Ciśnienie w cieczy przenosi się bez strat, więc mała siła przyłożona do wąskiego tłoka wywołuje proporcjonalnie dużo większą siłę na szerokim tłoku wyjściowym.",
        "followup_q": "Dlaczego w górach woda wrze w niższej temperaturze?",
        "followup_a": "Bo wraz z wysokością spada ciśnienie atmosferyczne, więc prężność pary wodnej zrównuje się z ciśnieniem otoczenia już poniżej 100°C.",
    },
    {
        "domain": "fizyka",
        "title": "Fuzja i rozszczepienie jądrowe",
        "fact": "Rozszczepienie dzieli ciężkie jądra atomowe (np. uran-235), natomiast fuzja termojądrowa łączy lekkie jądra wodoru w hel, zasilając gwiazdy.",
        "q": "Czym różni się fuzja jądrowa od rozszczepienia?",
        "a": "Rozszczepienie rozbija ciężkie jądra jak uran czy pluton (tak działają obecne elektrownie atomowe), a fuzja scala lekkie jądra wodoru w hel, uwalniając jeszcze więcej energii na jednostkę masy.",
        "followup_q": "Skąd w fuzji bierze się energia?",
        "followup_a": "Z tzw. deficytu masy — powstałe jądro helu ma nieco mniejszą masę niż wyjściowe protony i neutrony, a różnica zamienia się w energię według wzoru E = m * c^2.",
    },

    # ── 2. Chemia (10) ───────────────────────────────────────────────────────
    {
        "domain": "chemia",
        "title": "Układ okresowy pierwiastków Mendelejewa",
        "fact": "Układ okresowy porządkuje pierwiastki według rosnącej liczby atomowej (liczby protonów w jądrze), grupując pierwiastki o podobnej budowie powłoki walencyjnej.",
        "q": "Co decyduje o miejscu pierwiastka w układzie okresowym?",
        "a": "O miejscu pierwiastka decyduje jego liczba atomowa (liczba protonów w jądrze), natomiast numer grupy odzwierciedla liczbę elektronów walencyjnych i podobne właściwości chemiczne.",
        "followup_q": "Które pierwiastki tworzą grupę gazów szlachetnych?",
        "followup_a": "Hel, neon, argon, krypton, ksenon i radon — mają zapełnioną zewnętrzną powłokę elektronową, dlatego są niezwykle mało reaktywne.",
    },
    {
        "domain": "chemia",
        "title": "Budowa cząsteczki wody i wiązania wodorowe",
        "fact": "Cząsteczka wody (H2O) jest polarna o kształcie kątowym około 104,5°, co pozwala na tworzenie silnych wiązań wodorowych między cząsteczkami.",
        "q": "Dlaczego lód unosi się na powierzchni wody?",
        "a": "Wiązania wodorowe wymuszają w krysztale lodu luźną sieć przestrzenną o większej objętości, przez co lód ma mniejszą gęstość niż ciekła woda i wypływa na wierzch.",
        "followup_q": "Jakie znaczenie ma to dla życia w jeziorach zimą?",
        "followup_a": "Zamarzająca od góry warstwa lodu izoluje głębszą wodę przed wyziębieniem, dzięki czemu organizmy wodne mogą przetrwać zimę pod lodem.",
    },
    {
        "domain": "chemia",
        "title": "Skala pH, kwasy i zasady",
        "fact": "Skala pH mierzy stężenie jonów wodorowych w roztworze wodnym: poniżej 7 oznacza odczyn kwasowy, 7 obojętny, a powyżej 7 zasadowy.",
        "q": "Co oznacza odczyn pH poniżej 7?",
        "a": "Roztwór o pH mniejszym niż 7 jest kwasowy i zawiera nadmiar jonów wodorowych (H3O+) względem jonów wodorotlenkowych (OH-).",
        "followup_q": "Co powstaje w reakcji zobojętniania kwasu zasadą?",
        "followup_a": "Powstaje sól oraz woda, a odczyn roztworu zbliża się do obojętnego.",
    },
    {
        "domain": "chemia",
        "title": "Węgiel i jego odmiany alotropowe",
        "fact": "Węgiel tworzy miliony związków organicznych dzięki zdolności łączenia się w łańcuchy, a jako czysty pierwiastek występuje m.in. jako diament, grafit i grafen.",
        "q": "Dlaczego diament i grafit mają zupełnie inne właściwości, choć oba są z węgla?",
        "a": "W diamencie każdy atom węgla wiąże się przestrzennie z czterema sąsiadami tworząc najtwardszą sieć krystaliczną, a w graficie układa się w płaskie warstwy łatwo przesuwające się względem siebie.",
        "followup_q": "Czym na ich tle jest grafen?",
        "followup_a": "Grafen to pojedyncza, dwuwymiarowa warstwa atomów węgla ułożona w sześciokąty o wyjątkowej wytrzymałości i świetnym przewodnictwie elektrycznym.",
    },
    {
        "domain": "chemia",
        "title": "Reakcje utleniania i redukcji (redoks)",
        "fact": "W reakcjach redoks jeden reagent oddaje elektrony (utlenianie), a drugi je przyjmuje (redukcja) — oba procesy zachodzą zawsze równocześnie.",
        "q": "Na czym polega reakcja redoks w chemii?",
        "a": "Reakcja redoks to jednoczesne utlenianie (oddanie elektronów i wzrost stopnia utlenienia) oraz redukcja (przyjęcie elektronów i obniżenie stopnia utlenienia).",
        "followup_q": "Podaj codzienny przykład takiej reakcji.",
        "followup_a": "Rdzewienie żelaza, spalanie paliw oraz praca akumulatora litowo-jonowego w telefonie.",
    },
    {
        "domain": "chemia",
        "title": "Katalizatory i szybkość reakcji chemicznych",
        "fact": "Katalizator przyspiesza reakcję chemiczną poprzez obniżenie energii aktywacji, samemu nie zużywając się w końcowym bilansie procesu.",
        "q": "Jak działa katalizator w reakcji chemicznej?",
        "a": "Otwiera alternatywną ścieżkę reakcji o niższej barierze energetycznej (energii aktywacji), dzięki czemu reakcja biegnie szybciej i odtwarza katalizator na końcu.",
        "followup_q": "Jak nazywają się katalizatory w organizmach żywych?",
        "followup_a": "W biologii funkcję precyzyjnych katalizatorów pełnią enzymy — najczęściej wyspecjalizowane białka.",
    },
    {
        "domain": "chemia",
        "title": "Wiązania jonowe i kowalencyjne",
        "fact": "Wiązanie kowalencyjne powstaje przez uwspólnienie pary elektronów między niemetalami, a jonowe przez przyciąganie elektrostatyczne kationów i anionów.",
        "q": "Czym różni się wiązanie kowalencyjne od jonowego?",
        "a": "W wiązaniu kowalencyjnym atomy dzielą się elektronami (np. w cząsteczce O2 lub CO2), natomiast w jonowym jeden atom przekazuje elektron drugiemu, tworząc przeciwne jony (np. Na+ i Cl- w soli).",
        "followup_q": "Dlaczego roztwór soli kuchennej przewodzi prąd, a czysta woda prawie nie?",
        "followup_a": "Ponieważ po rozpuszczeniu kryształu NaCl w wodzie uwalniają się ruchliwe jony Na+ i Cl-, które przenoszą ładunek elektryczny.",
    },
    {
        "domain": "chemia",
        "title": "Skład powietrza atmosferycznego",
        "fact": "Suche powietrze przy powierzchni Ziemi składa się w około 78% z azotu (N2), 21% z tlenu (O2), 0,93% z argonu oraz około 0,04% z dwutlenku węgla.",
        "q": "Jakiego gazu jest najwięcej w ziemskiej atmosferze?",
        "a": "Najwięcej jest azotu — stanowi aż około 78% objętości suchego powietrza, podczas gdy tlen to około 21%, a argon niemal 1%.",
        "followup_q": "Jaką rolę pełni niewielki udział pary wodnej i CO2?",
        "followup_a": "Pochłaniają promieniowanie podczerwone emitowane przez powierzchnię Ziemi, tworząc naturalny efekt cieplarniany utrzymujący dodatnią średnią temperaturę planety.",
    },
    {
        "domain": "chemia",
        "title": "Polimery naturalne i syntetyczne",
        "fact": "Polimery to wielkocząsteczkowe związki zbudowane z powtarzalnych merów — w przyrodzie są nimi m.in. celuloza, skrobia i białka, a w technice polietylen czy nylon.",
        "q": "Co to jest polimer i jak powstaje?",
        "a": "Polimer to bardzo długa cząsteczka złożona z tysięcy połączonych mniejszych jednostek zwanych monomerami w procesie polimeryzacji lub polikondensacji.",
        "followup_q": "Czy DNA też jest polimerem?",
        "followup_a": "Tak — DNA jest naturalnym biopolimerem zbudowanym z monomerów zwanych nukleotydami.",
    },
    {
        "domain": "chemia",
        "title": "Promieniotwórczość i czas połowicznego rozpadu",
        "fact": "Czas połowicznego rozpadu (T1/2) to okres, po którym statystycznie rozpada się połowa jąder danego izotopu promieniotwórczego.",
        "q": "Do czego w archeologii wykorzystuje się izotop węgla C-14?",
        "a": "Datowanie radiowęglowe mierzy ubytek promieniotwórczego izotopu C-14 (czas połowicznego rozpadu ok. 5730 lat), co pozwala określić wiek szczątków organicznych sprzed nawet kilkudziesięciu tysięcy lat.",
        "followup_q": "Dlaczego metoda C-14 nie działa na skałach sprzed milionów lat?",
        "followup_a": "Bo po kilkunastu okresach półtrwania niemal cały węgiel C-14 ulega rozpadowi i do starszych skał stosuje się izotopy o znacznie dłuższym okresie półtrwania, np. uran-ołów.",
    },

    # ── 3. Biologia i Medycyna (11) ──────────────────────────────────────────
    {
        "domain": "biologia_medycyna",
        "title": "Budowa i funkcja DNA",
        "fact": "DNA ma strukturę podwójnej helisy, w której cztery zasady azotowe (A z T oraz C z G) kodują instrukcję syntezy białek w każdej żywej komórce.",
        "q": "Jak kodowana jest informacja genetyczna w cząsteczce DNA?",
        "a": "Informacja zapisana jest w kolejności nukleotydów oznaczanych literami A, T, C i G — trójki zasad (kodony) wyznaczają kolejne aminokwasy podczas budowy białek.",
        "followup_q": "Czym różni się RNA od DNA?",
        "followup_a": "RNA jest zazwyczaj jednoniciowe, zawiera rybozę zamiast deoksyrybozy oraz uracyl (U) zamiast tyminy (T), pełniąc rolę posłańca i maszynerii przy produkcji białek.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Fotosynteza i chloroplasty",
        "fact": "Podczas fotosyntezy rośliny wykorzystują energię światła i chlorofil do przekształcenia dwutlenku węgla oraz wody w glukozę i tlen.",
        "q": "Co jest potrzebne roślinom do przeprowadzenia fotosyntezy?",
        "a": "Rośliny potrzebują światła słonecznego, wody pobieranej przez korzenie, dwutlenku węgla z powietrza oraz zielonego barwnika — chlorofilu w chloroplastach.",
        "followup_q": "Co powstaje jako produkt uboczny tego procesu?",
        "followup_a": "Tlen atmosferyczny (O2), uwalniany do powietrza przez aparaty szparkowe w liściach.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Mitochondria i ATP — energetyka komórki",
        "fact": "Mitochondria produkują większość komórkowego ATP w procesie oddychania tlenowego, dlatego nazywa się je elektrowniami komórki.",
        "q": "Dlaczego mitochondria nazywamy elektrowniami komórki?",
        "a": "Ponieważ w procesie fosforylacji oksydacyjnej wytwarzają cząsteczki ATP — uniwersalny nośnik energii chemicznej napędzający skurcze mięśni, transport i syntezę białek.",
        "followup_q": "Co ciekawego wyróżnia DNA mitochondrialne?",
        "followup_a": "Mitochondria posiadają własny, krótki kolisty genom (mtDNA), który u ludzi dziedziczy się praktycznie wyłącznie po matce.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Układ krwionośny i hemoglobina",
        "fact": "Serce pompuje krew przez mały obieg płucny w celu natlenowania i przez duży obieg ustrojowy do wszystkich tkanek, gdzie tlen przenosi hemoglobina w erytrocytach.",
        "q": "Jaką rolę w organizmie pełnią czerwone krwinki?",
        "a": "Erytrocyty zawierają białko hemoglobinę z atomem żelaza, które wiąże tlen w pęcherzykach płucnych i dostarcza go do wszystkich komórek ciała.",
        "followup_q": "A za co odpowiadają białe krwinki i płytki krwi?",
        "followup_a": "Leukocyty (białe krwinki) zwalczają infekcje w układzie odpornościowym, natomiast trombocyty (płytki krwi) odpowiadają za krzepnięcie i tamowanie krwawień.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Synapsa i neuroprzekaźniki w mózgu",
        "fact": "Neurony komunikują się przez synapsy, gdzie impuls elektryczny wyzwala neuroprzekaźniki (np. glutaminian, GABA, dopaminę) pobudzające lub hamujące kolejny neuron.",
        "q": "Jak jeden neuron przekazuje sygnał drugiemu?",
        "a": "Po dotarciu potencjału czynnościowego do zakończenia aksonu uwalniane są neuroprzekaźniki do szczeliny synaptycznej, gdzie wiążą się z receptorami kolejnego neuronu.",
        "followup_q": "Który neuroprzekaźnik głównie pobudza, a który hamuje korę mózgową?",
        "followup_a": "Głównym neuroprzekaźnikiem pobudzającym jest glutaminian, natomiast najważniejszym hamującym w mózgu jest GABA.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Układ odpornościowy i szczepienia",
        "fact": "Układ odpornościowy dzieli się na wrodzony (szybka odpowiedź nieswoista) i nabyty (limfocyty T i B produkujące przeciwciała oraz pamięć immunologiczną).",
        "q": "W jaki sposób szczepionka chroni przed chorobą?",
        "a": "Szczepionka uczy układ odpornościowy rozpoznawania bezpiecznych fragmentów lub osłabionego patogenu, wytwarzając komórki pamięci zdolne błyskawicznie zniszczyć prawdziwego wirusa lub bakterię.",
        "followup_q": "Czym antybiotyk różni się od szczepionki?",
        "followup_a": "Antybiotyk leczy już trwające zakażenia bakteryjne (nie działa na wirusy!), podczas gdy szczepionka zapobiega zakażeniu zawczasu.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Ewolucja przez dobór naturalny",
        "fact": "Według teorii Darwina losowe mutacje dające przewagę w przetrwaniu i rozmnażaniu utrwalają się w kolejnych pokoleniach w procesie doboru naturalnego.",
        "q": "Na czym polega mechanizm doboru naturalnego?",
        "a": "Osobniki lepiej przystosowane do lokalnego środowiska częściej przeżywają i przekazują swoje geny potomstwu, co przez wiele pokoleń zmienia cechy całej populacji.",
        "followup_q": "Czy ewolucja ma zaplanowany cel?",
        "followup_a": "Nie — mutacje zachodzą losowo, a środowisko na bieżąco selekcjonuje rozwiązania działające wystarczająco dobrze w danych warunkach.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Bakterie a wirusy",
        "fact": "Bakterie to samowystarczalne organizmy jednokomórkowe bez jądra komórkowego, natomiast wirusy to bezkomórkowe pakiety kwasu nukleinowego wymagające komórki gospodarza do replikacji.",
        "q": "Dlaczego antybiotyki nie leczą grypy ani przeziębienia?",
        "a": "Ponieważ grypa i większość przeziębień wywołują wirusy, a antybiotyki niszczą struktury typowe wyłącznie dla bakterii, takie jak ściana komórkowa czy bakteryjne rybosomy.",
        "followup_q": "Czy wszystkie bakterie w ciele człowieka są szkodliwe?",
        "followup_a": "Wręcz przeciwnie — miliardy pożytecznych bakterii mikrobioty jelitowej wspomagają trawienie, produkują witaminy (m.in. K i B) oraz blokują rozwój patogenów.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Fazy snu NREM i REM",
        "fact": "Sen człowieka przebiega cyklicznie przez głęboki sen NREM (regeneracja ciała i konsolidacja pamięci deklaratywnej) oraz fazę REM (marzenia senne i integracja skojarzeń).",
        "q": "Jaką funkcję pełni głęboki sen NREM dla mózgu?",
        "a": "W głębokim śnie NREM hipokamp odtwarza świeże wspomnienia do kory mózgowej, a układ glimfatyczny oczyszcza tkankę mózgową z produktów przemiany materii.",
        "followup_q": "Co natomiast dzieje się w fazie REM?",
        "followup_a": "Aktywność kory wzrasta, pojawiają się żywe marzenia senne, następuje przejściowe zwiotczenie mięśni, a mózg łączy odległe wspomnienia i porządkuje emocje.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "Mikroskopowe działanie nerek i wątroby",
        "fact": "Wątroba metabolizuje składniki odżywcze, magazynuje glikogen i neutralizuje toksyny, natomiast nerki filtrują krew w nefronach, regulując wodę i elektrolity.",
        "q": "Jaka jest główna różnica między rolą wątroby a nerek?",
        "a": "Wątroba jest głównym laboratorium chemicznym organizmu (rozkłada leki, alkohol i produkuje żółć), a nerki mechanicznie i osmotycznie filtrują osocze, usuwając mocznik oraz nadmiar wody.",
        "followup_q": "Czy wątroba potrafi się regenerować?",
        "followup_a": "Tak — wątroba ma wyjątkową wśród narządów wewnętrznych zdolność odbudowy swojej masy nawet po utracie znacznej części tkanki.",
    },
    {
        "domain": "biologia_medycyna",
        "title": "CRISPR-Cas9 i edycja genów",
        "fact": "Technologia CRISPR-Cas9 pochodzi z bakteryjnego systemu obrony przed wirusami i pozwala precyzyjnie ciąć oraz modyfikować wybrane sekwencje DNA.",
        "q": "Czym w biotechnologii jest narzędzie CRISPR-Cas9?",
        "a": "To precyzyjne „nożyczki molekularne”, w których krótki przewodnik RNA wskazuje białku Cas9 dokładne miejsce w DNA do przecięcia lub naprawy wadliwego genu.",
        "followup_q": "Jakie choroby pomaga zwalczać?",
        "followup_a": "Zatwierdzono już pierwsze terapie genowe oparte na CRISPR w leczeniu anemii sierpowatej i beta-talasemii oraz prowadzi się badania nad chorobami siatkówki i nowotworami.",
    },

    # ── 4. Astronomia i Kosmos (10) ──────────────────────────────────────────
    {
        "domain": "astronomia",
        "title": "Układ Słoneczny i kolejność planet",
        "fact": "W Układzie Słonecznym krąży osiem planet w kolejności od Słońca: Merkury, Wenus, Ziemia, Mars (skaliste) oraz Jowisz, Saturn, Uran i Neptun (gazowe i lodowe olbrzymy).",
        "q": "Wymień planety Układu Słonecznego po kolei od Słońca.",
        "a": "Kolejno od Słońca są to: Merkury, Wenus, Ziemia, Mars, Jowisz, Saturn, Uran oraz Neptun.",
        "followup_q": "Która planeta jest najgorętsza i dlaczego nie jest to Merkury?",
        "followup_a": "Najgorętsza jest Wenus (ok. 465°C), ponieważ jej gęsta atmosfera z dwutlenku węgla wywołuje ekstremalny efekt cieplarniany.",
    },
    {
        "domain": "astronomia",
        "title": "Pory roku i nachylenie osi Ziemi",
        "fact": "Pory roku nie wynikają ze zmiennej odległości od Słońca, lecz z nachylenia osi obrotu Ziemi o około 23,4° względem płaszczyzny jej orbity.",
        "q": "Skąd biorą się pory roku na Ziemi?",
        "a": "Wynikają z nachylenia osi ziemskiej o ok. 23,4° — gdy półkula północna jest nachylona ku Słońcu, dni są dłuższe i promienie padają bardziej stromo, dając lato.",
        "followup_q": "Kiedy Ziemia jest fizycznie najbliżej Słońca?",
        "followup_a": "W peryhelium na początku stycznia — czyli w środku północnej zimy, co dowodzi, że decyduje kąt padania promieni, a nie odległość.",
    },
    {
        "domain": "astronomia",
        "title": "Czarne dziury i horyzont zdarzeń",
        "fact": "Czarna dziura to obszar czasoprzestrzeni o tak silnej grawitacji, że spoza jej granicy — horyzontu zdarzeń — nie może uciec nawet światło.",
        "q": "Co to jest horyzont zdarzeń czarnej dziury?",
        "a": "Horyzont zdarzeń to granica bez powrotu wokół czarnej dziury: prędkość ucieczki przekracza tam prędkość światła, więc żaden sygnał z wnętrza nie wydostaje się na zewnątrz.",
        "followup_q": "Jak astronomowie mogą obserwować czarne dziury, skoro nie świecą?",
        "followup_a": "Obserwują rozgrzaną materię w dysku akrecyjnym przed horyzontem, ruch pobliskich gwiazd oraz fale grawitacyjne ze zderzeń czarnych dziur.",
    },
    {
        "domain": "astronomia",
        "title": "Rok świetlny jako miara odległości",
        "fact": "Rok świetlny to odległość, jaką światło przebywa w próżni w ciągu jednego roku juliańskiego — wynosi około 9,46 biliona kilometrów.",
        "q": "Czy rok świetlny jest jednostką czasu, czy odległości?",
        "a": "Rok świetlny jest jednostką odległości (ok. 9,46 × 10^12 km) — oznacza dystans, jaki pokonuje promień światła przez jeden rok.",
        "followup_q": "Jak daleko leży najbliższa po Słońcu gwiazda Proxima Centauri?",
        "followup_a": "Około 4,24 roku świetlnego od Ziemi, co oznacza, że jej światło leci do nas ponad cztery lata.",
    },
    {
        "domain": "astronomia",
        "title": "Wielki Wybuch i mikrofalowe promieniowanie tła",
        "fact": "Wszechświat rozszerza się od około 13,8 miliarda lat od stanu skrajnie gęstego i gorącego, a echem wczesnego kosmosu jest mikrofalowe promieniowanie tła.",
        "q": "Skąd wiemy, że Wszechświat miał gorący początek?",
        "a": "Potwierdzają to trzy główne obserwacje: ucieczka galaktyk (prawo Hubble'a), jednorodne mikrofalowe promieniowanie tła z każdego kierunku oraz obfitość pierwotnego wodoru i helu.",
        "followup_q": "Ile lat ma obecnie Wszechświat?",
        "followup_a": "Wiek Wszechświata szacuje się na około 13,8 miliarda lat.",
    },
    {
        "domain": "astronomia",
        "title": "Fazy Księżyca i pływy morskie",
        "fact": "Fazy Księżyca zależą od kąta jego oświetlenia przez Słońce widzianego z Ziemi, a przyciąganie grawitacyjne Księżyca i Słońca wywołuje pływy oceaniczne.",
        "q": "Dlaczego zawsze widzimy tę samą stronę Księżyca?",
        "a": "Księżyc znajduje się w rotacji synchronicznej — obraca się wokół własnej osi w dokładnie tym samym czasie (ok. 27,3 dnia), w jakim okrąża Ziemię.",
        "followup_q": "Kiedy na oceanach występują największe pływy (syzygijne)?",
        "followup_a": "Podczas nowiu i pełni, gdy Słońce, Ziemia i Księżyc ustawiają się w jednej linii i ich siły pływowe sumują się.",
    },
    {
        "domain": "astronomia",
        "title": "Droga Mleczna i inne galaktyki",
        "fact": "Układ Słoneczny leży w jednym ze spiralnych ramion galaktyki Drogi Mlecznej, liczącej szacunkowo od 100 do 400 miliardów gwiazd.",
        "q": "Jaka duża galaktyka spiralna znajduje się najbliżej Drogi Mlecznej?",
        "a": "Najbliższą dużą galaktyką spiralną jest Galaktyka Andromedy (M31), oddalona o około 2,5 miliona lat świetlnych.",
        "followup_q": "Co wydarzy się z nimi za kilka miliardów lat?",
        "followup_a": "Droga Mleczna i Andromeda zbliżają się do siebie i za około 4,5 miliarda lat połączą się w jedną wielką galaktykę eliptyczną.",
    },
    {
        "domain": "astronomia",
        "title": "Supernowe i pochodzenie ciężkich pierwiastków",
        "fact": "Pierwiastki cięższe od helu, takie jak węgiel, tlen czy żelazo, powstają w jądrach masywnych gwiazd oraz podczas wybuchów supernowych i zderzeń gwiazd neutronowych.",
        "q": "Skąd wzięło się żelazo we krwi i złoto w biżuterii na Ziemi?",
        "a": "Żelazo oraz tlen wykuły się we wnętrzach dawnych gwiazd i zostały rozsiane przez wybuchy supernowych, a najcięższe pierwiastki (np. złoto, platyna) powstały głównie w zderzeniach gwiazd neutronowych.",
        "followup_q": "Co może pozostać po wybuchu supernowej?",
        "followup_a": "W zależności od masy jądra gwiazdy powstaje gęsta gwiazda neutronowa (pulsar) albo czarna dziura.",
    },
    {
        "domain": "astronomia",
        "title": "Teleskop Jamesa Webba i obserwacje w podczerwieni",
        "fact": "Kosmiczny Teleskop Jamesa Webba (JWST) pracuje w punkcie L2 w odległości 1,5 mln km od Ziemi i obserwuje Wszechświat głównie w podczerwieni.",
        "q": "Dlaczego Teleskop Jamesa Webba obserwuje kosmos w podczerwieni?",
        "a": "Światło najstarszych galaktyk zostało rozciągnięte ku podczerwieni przez rozszerzanie się Wszechświata, a dodatkowo podczerwień przenika przez obłoki pyłu zasłaniające narodziny gwiazd i egzoplanety.",
        "followup_q": "Gdzie stacjonuje ten teleskop?",
        "followup_a": "Krąży wokół punktu libracyjnego L2 układu Słońce–Ziemia, około 1,5 miliona kilometrów za Ziemią.",
    },
    {
        "domain": "astronomia",
        "title": "Egzoplanety i strefa zamieszkiwalna",
        "fact": "Egzoplaneta to planeta krążąca wokół innej gwiazdy niż Słońce; w tzw. strefie zamieszkiwalnej na jej powierzchni może utrzymywać się woda w stanie ciekłym.",
        "q": "Jak najczęściej wykrywa się planety pozasłoneczne (egzoplanety)?",
        "a": "Najczęstszą metodą jest metoda tranzytu — astronomowie rejestrują cykliczne, minimalne przygasanie blasku gwiazdy, gdy planeta przechodzi na tle jej tarczy.",
        "followup_q": "Czy sama obecność w strefie zamieszkiwalnej gwarantuje życie?",
        "followup_a": "Nie — oprócz odpowiedniej odległości od gwiazdy potrzebna jest także odpowiednia atmosfera, pole magnetyczne oraz stabilny skład chemiczny.",
    },

    # ── 5. Geografia i Nauki o Ziemi (10) ────────────────────────────────────
    {
        "domain": "geografia",
        "title": "Tektonika płyt litosfery",
        "fact": "Zewnętrzna skorupa Ziemi dzieli się na poruszające się płyty tektoniczne, których zderzenia i rozsuwanie wywołują trzęsienia ziemi, wulkanizm i powstawanie gór.",
        "q": "Jak powstają wysokie pasma górskie takie jak Himalaje?",
        "a": "Himalaje wypiętrzyły się i nadal rosną na skutek powolnego zderzenia płyty indyjskiej z płytą euroazjatycką, które fałduje i spiętrza skorupę ziemską.",
        "followup_q": "Co dzieje się na granicach rozbieżnych płyt, np. na dnie Atlantyku?",
        "followup_a": "Tam płyty oddalają się od siebie, a wypływająca magma tworzy nowy grzbiet oceaniczny i poszerza dno morskie.",
    },
    {
        "domain": "geografia",
        "title": "Polska — geografia, rzeki i góry",
        "fact": "Najdłuższą rzeką Polski jest Wisła (1047 km), najwyższym szczytem Rysy w Tatrach (2499 m n.p.m.), a powierzchnia kraju wynosi około 312,7 tys. km².",
        "q": "Jaka jest najdłuższa rzeka Polski i najwyższy szczyt w kraju?",
        "a": "Najdłuższą rzeką Polski jest Wisła (1047 km) wpadająca do Morza Bałtyckiego, a najwyższym szczytem leżącym w granicach Polski jest północno-zachodni wierzchołek Rysów w Tatrach (2499 m n.p.m.).",
        "followup_q": "Z iloma państwami graniczy Polska na lądzie?",
        "followup_a": "Polska graniczy z siedmioma państwami: Niemcami, Czechami, Słowacją, Ukrainą, Białorusią, Litwą i Rosją (obwodem królewieckim).",
    },
    {
        "domain": "geografia",
        "title": "Najwyższy szczyt i najgłębszy rów Ziemi",
        "fact": "Mount Everest (Czomolungma) w Himalajach wznosi się na 8848,86 m n.p.m., a Rów Mariański na Pacyfiku osiąga w Głębi Challengera około 10 920 m p.p.m.",
        "q": "Gdzie znajduje się najgłębsze znane miejsce w oceanach Ziemi?",
        "a": "Najgłębsze miejsce to Głębia Challengera w Rowie Mariańskim na zachodnim Oceanie Spokojnym, osiągająca niemal 11 kilometrów głębokości.",
        "followup_q": "Który szczyt jest najwyższy od poziomu morza?",
        "followup_a": "Mount Everest na granicy Nepalu i Chin o wysokości około 8849 metrów nad poziomem morza.",
    },
    {
        "domain": "geografia",
        "title": "Obieg wody w przyrodzie (cykl hydrologiczny)",
        "fact": "Słońce napędza obieg wody: parowanie z oceanów i lądów, kondensację w chmurach, opady atmosferyczne oraz spływ powierzchniowy i podziemny z powrotem do mórz.",
        "q": "Co napędza nieustanny obieg wody na Ziemi?",
        "a": "Głównym motorem cyklu hydrologicznego jest energia słoneczna (powodująca parowanie i transpirację roślin) oraz grawitacja ściągająca opady i rzeki ku oceanom.",
        "followup_q": "Ile procent wody na Ziemi stanowi woda słodka?",
        "followup_a": "Zaledwie około 2,5% — z czego większość jest uwięziona w lodowcach i lądolodach, a tylko ułamek procenta płynie w rzekach i jeziorach.",
    },
    {
        "domain": "geografia",
        "title": "Prądy morskie i Golfsztrom",
        "fact": "Prąd Zatokowy (Golfsztrom) niesie ciepłe wody z Zatoki Meksykańskiej ku północnemu Atlantykowi, wyraźnie łagodząc klimat Europy Zachodniej i Północnej.",
        "q": "Dlaczego w Europie Północnej jest cieplej niż na tej samej szerokości geograficznej w Kanadzie?",
        "a": "Ponieważ ciepły Prąd Zatokowy i Prąd Północnoatlantycki przynoszą z południa olbrzymie ilości ciepła oceanicznego, ogrzewając wiatry napływające nad Europę.",
        "followup_q": "Jakie jest zjawisko pogodowe El Niño?",
        "followup_a": "El Niño to okresowe ocieplenie wód równikowego Pacyfiku, które zmienia układ wiatrów i opadów na niemal całym globie.",
    },
    {
        "domain": "geografia",
        "title": "Warstwy atmosfery ziemskiej",
        "fact": "Atmosfera dzieli się od dołu na troposferę (gdzie zachodzi pogoda), stratosferę (z warstwą ozonową), mezosferę, termosferę oraz egzosferę.",
        "q": "W której warstwie atmosfery powstają chmury i pogoda?",
        "a": "Prawie cała pogoda, chmury i wiatry powstają w najniższej warstwie — troposferze, sięgającej od powierzchni Ziemi do około 8–17 km wysokości.",
        "followup_q": "Co znajduje się wyżej w stratosferze?",
        "followup_a": "W stratosferze leży warstwa ozonowa pochłaniająca szkodliwe promieniowanie ultrafioletowe Słońca (UV-B i UV-C).",
    },
    {
        "domain": "geografia",
        "title": "Największe rzeki i kontynenty świata",
        "fact": "Na Ziemi wyróżnia się 7 kontynentów, z których największa jest Azja; najdłuższymi rzekami świata są Nil i Amazonka niosąca najwięcej wody.",
        "q": "Która rzeka na świecie niesie najwięcej wody?",
        "a": "Zdecydowanie najzasobniejsza w wodę jest Amazonka w Ameryce Południowej — jej średni przepływ jest wielokrotnie większy niż kolejnych wielkich rzek świata.",
        "followup_q": "A który kontynent jest największy pod względem powierzchni i ludności?",
        "followup_a": "Azja — obejmuje około jednej trzeciej powierzchni lądów i mieszka na niej ponad połowa ludności Ziemi.",
    },
    {
        "domain": "geografia",
        "title": "Strefy czasowe i linia zmiany daty",
        "fact": "Ziemia obraca się o 360° w ciągu 24 godzin, czyli o 15° na godzinę, co stanowi podstawę podziału globu na 24 podstawowe strefy czasowe.",
        "q": "Dlaczego na świecie wprowadzono strefy czasowe?",
        "a": "Ponieważ w danym momencie południe słoneczne występuje tylko na jednym południku — podział na strefy co ok. 15° długości geograficznej pozwala utrzymać podobną porę dnia w każdym regionie.",
        "followup_q": "Gdzie przebiega międzynarodowa linia zmiany daty?",
        "followup_a": "Przebiega głównie wzdłuż południka 180° na Oceanie Spokojnym, omijając zamieszkane wyspy.",
    },
    {
        "domain": "geografia",
        "title": "Morze Bałtyckie — cechy charakterystyczne",
        "fact": "Morze Bałtyckie jest młodym, płytkim morzem śródlądowym o niskim zasoleniu (średnio ok. 7 promili) z ograniczoną wymianą wód przez Cieśniny Duńskie.",
        "q": "Dlaczego Bałtyk jest znacznie mniej słony niż oceany?",
        "a": "Ponieważ wpada do niego ponad dwieście pięćdziesiąt rzek niosących słodką wodę, a wąskie i płytkie Cieśniny Duńskie ograniczają dopływ słonych wód z Morza Północnego.",
        "followup_q": "Jaka jest średnia głębokość Morza Bałtyckiego?",
        "followup_a": "Średnia głębokość wynosi zaledwie około 52 metry (maksymalna w Głębi Landsort to 459 m), przez co Bałtyk jest bardzo wrażliwy na zanieczyszczenia.",
    },
    {
        "domain": "geografia",
        "title": "Wulkany i Pacyficzny Pierścień Ognia",
        "fact": "Około trzech czwartych aktywnych wulkanów lądowych Ziemi leży wokół Oceanu Spokojnego w tzw. Pacyficznym Pierścieniu Ognia na strefach subdukcji płyt.",
        "q": "Dlaczego wokół Oceanu Spokojnego jest tak wiele wulkanów i trzęsień ziemi?",
        "a": "Ponieważ wzdłuż brzegów Pacyfiku ciężkie płyty oceaniczne podsuwają się pod płyty kontynentalne (subdukcja), topiąc skały w głębi płaszcza i generując częste wstrząsy oraz erupcje.",
        "followup_q": "Czym magma różni się od lawy?",
        "followup_a": "Magma to stopiona skała znajdująca się pod powierzchnią Ziemi, a lawą nazywamy ją po wydostaniu się na powierzchnię.",
    },

    # ── 6. Historia i Cywilizacja (10) ───────────────────────────────────────
    {
        "domain": "historia",
        "title": "Chrzest Polski i początki państwowości (966)",
        "fact": "Chrzest księcia Mieszka I w 966 roku włączył młode państwo Polan w krąg chrześcijańskiej kultury łacińskiej i wzmocnił jego pozycję dyplomatyczną w Europie.",
        "q": "Jakie znaczenie miał chrzest Polski w 966 roku?",
        "a": "Chrzest Mieszka I w 966 roku wprowadził Polskę do wspólnoty państw chrześcijańskich Europy, ułatwił sojusz z Czechami, rozwój piśmiennictwa oraz utworzenie własnej organizacji kościelnej.",
        "followup_q": "Kto został pierwszym koronowanym królem Polski?",
        "followup_a": "Syn Mieszka I — Bolesław I Chrobry, który koronował się w 1025 roku.",
    },
    {
        "domain": "historia",
        "title": "Bitwa pod Grunwaldem (1410)",
        "fact": "15 lipca 1410 roku połączone wojska polsko-litewskie pod dowództwem króla Władysława Jagiełły pokonały zakon krzyżacki pod Grunwaldem.",
        "q": "Kto wygrał bitwę pod Grunwaldem i kiedy się odbyła?",
        "a": "Bitwa odbyła się 15 lipca 1410 roku i zakończyła się zdecydowanym zwycięstwem wojsk Korony Królestwa Polskiego oraz Wielkiego Księstwa Litewskiego pod wodzą Władysława Jagiełły nad zakonem krzyżackim.",
        "followup_q": "Kto dowodził stroną litewską i zakonem krzyżackim?",
        "followup_a": "Wojskami litewskimi dowodził wielki książę Witold, a zakonem krzyżackim wielki mistrz Ulrich von Jungingen, który poległ w bitwie.",
    },
    {
        "domain": "historia",
        "title": "Rewolucja druku Gutenberga (XV wiek)",
        "fact": "Wynalezienie ruchomej czcionki odlewanej z metalu i prasy drukarskiej przez Jana Gutenberga około 1450 roku gwałtownie obniżyło koszt książek i przyspieszyło renesans oraz naukę.",
        "q": "Jak wynalazek Gutenberga zmienił Europę?",
        "a": "Ruchoma czcionka i prasa drukarska umożliwiły masowe i tanie powielanie książek, co błyskawicznie upowszechniło umiejętność czytania, idee renesansu, reformacji oraz odkrycia naukowe.",
        "followup_q": "Gdzie wynaleziono papier i wcześniejsze formy druku?",
        "followup_a": "Zarówno papier, jak i wczesny druk drzeworytowy oraz pierwsze czcionki narodziły się wcześniej w Chinach i Korei, lecz europejski stop metali Gutenberga zrewolucjonizował druk masowy.",
    },
    {
        "domain": "historia",
        "title": "Konstytucja 3 maja (1791)",
        "fact": "Uchwalona 3 maja 1791 roku przez Sejm Wielki ustawa rządowa Rzeczypospolitej była pierwszą nowoczesną konstytucją w Europie i drugą na świecie po amerykańskiej.",
        "q": "Dlaczego Konstytucja 3 maja 1791 roku jest tak ważna w historii?",
        "a": "Była pierwszą w Europie i drugą na świecie spisaną ustawą zasadniczą nowoczesnego państwa: zniosła liberum veto, wprowadziła trójpodział władzy i wzięła pod opiekę prawa chłopów oraz mieszczan.",
        "followup_q": "Co niestety przeszkodziło w jej pełnym wdrożeniu?",
        "followup_a": "Konfederacja targowicka oraz interwencja zbrojna Rosji w 1792 roku, które doprowadziły do drugiego, a następnie trzeciego rozbioru Polski.",
    },
    {
        "domain": "historia",
        "title": "Rewolucja przemysłowa",
        "fact": "Zapoczątkowana w XVIII wieku w Wielkiej Brytanii rewolucja przemysłowa zastąpiła pracę ręczną maszynami parowymi, fabrykami, koleją i masową produkcją stali.",
        "q": "Jakie wynalazki napędzały pierwszą rewolucję przemysłową?",
        "a": "Kluczowe było ulepszenie maszyny parowej przez Jamesa Watta, mechanizacja przędzalnictwa i tkactwa, rozwój hutnictwa na koksie oraz budowa pierwszych linii kolejowych.",
        "followup_q": "Co przyniosła druga rewolucja przemysłowa pod koniec XIX wieku?",
        "followup_a": "Elektryfikację, silnik spalinowy, telefon, telegraf, rozwój przemysłu chemicznego oraz taśmową produkcję seryjną.",
    },
    {
        "domain": "historia",
        "title": "Odzyskanie niepodległości przez Polskę (1918)",
        "fact": "11 listopada 1918 roku po zakończeniu I wojny światowej i 123 latach zaborów Polska odrodziła się jako niepodległa II Rzeczpospolita.",
        "q": "Kiedy Polska odzyskała niepodległość po rozbiorach?",
        "a": "Po 123 latach zaborów Polska odzyskała niepodległość w listopadzie 1918 roku — symbolicznie obchodzimy to 11 listopada, gdy Józef Piłsudski przejął władzę wojskową w Warszawie.",
        "followup_q": "Kogo zalicza się do głównych Ojców Niepodległości?",
        "followup_a": "Wymienia się najczęściej Józefa Piłsudskiego, Romana Dmowskiego, Ignacego Jana Paderewskiego, Wincentego Witosa, Wojciecha Korfantego i Ignacego Daszyńskiego.",
    },
    {
        "domain": "historia",
        "title": "Enigma i polscy kryptolodzy (1932)",
        "fact": "W 1932 roku polscy matematycy Marian Rejewski, Jerzy Różycki i Henryk Zygalski złamali szyfr niemieckiej maszyny Enigma metodami matematycznymi.",
        "q": "Kto jako pierwszy złamał kod niemieckiej maszyny Enigma?",
        "a": "Dokonali tego w 1932 roku polscy kryptolodzy Marian Rejewski, Jerzy Różycki i Henryk Zygalski, stosując teorię permutacji zamiast tradycyjnej analizy językowej.",
        "followup_q": "Co zrobiono z tą wiedzą przed wybuchem II wojny światowej?",
        "followup_a": "W lipcu 1939 roku Polska przekazała kopie maszyny Enigma i dokumentację sojusznikom w Wielkiej Brytanii oraz Francji, co stało się fundamentem prac zespołu Alana Turinga w Bletchley Park.",
    },
    {
        "domain": "historia",
        "title": "Maria Skłodowska-Curie i dwie Nagrody Nobla",
        "fact": "Maria Skłodowska-Curie odkryła polon i rad, ukuła termin „promieniotwórczość” i jako jedyna osoba w historii zdobyła Nagrody Nobla w dwóch różnych naukach ścisłych.",
        "q": "Z jakich osiągnięć znana jest Maria Skłodowska-Curie?",
        "a": "Odkryła dwa nowe pierwiastki (polon i rad), wyjaśniła zjawisko promieniotwórczości jako właściwość atomów i zdobyła dwie Nagrody Nobla: z fizyki (1903) oraz z chemii (1911).",
        "followup_q": "Jak pomagała podczas I wojny światowej?",
        "followup_a": "Zorganizowała mobilne ambulanse rentgenowskie zwane „małymi Curie”, które prześwietlały rannych żołnierzy bezpośrednio na froncie.",
    },
    {
        "domain": "historia",
        "title": "Program Apollo 11 (1969)",
        "fact": "20 lipca 1969 roku lądownik Eagle misji Apollo 11 osiadł na Księżycu, a Neil Armstrong i Buzz Aldrin jako pierwsi ludzie postawili stopę na innym ciele niebieskim.",
        "q": "Kto jako pierwszy postawił stopę na Księżycu?",
        "a": "Pierwszym człowiekiem na Księżycu był amerykański astronauta Neil Armstrong podczas misji Apollo 11 w lipcu 1969 roku; chwilę później dołączył do niego Buzz Aldrin.",
        "followup_q": "Kto pozostał wtedy na orbicie księżycowej?",
        "followup_a": "Michael Collins pilotował moduł dowodzenia Columbia na orbicie wokół Księżyca, czekając na powrót kolegów.",
    },
    {
        "domain": "historia",
        "title": "Upadek muru berlińskiego i transformacja 1989",
        "fact": "W 1989 roku wybory czerwcowe i powołanie rządu Tadeusza Mazowieckiego w Polsce zapoczątkowały Jesień Ludów w Europie Środkowej oraz upadek muru berlińskiego.",
        "q": "Co zapoczątkowało upadek komunizmu w Europie Środkowej w 1989 roku?",
        "a": "Kluczowy impuls dały obrady Okrągłego Stołu i częściowo wolne wybory w Polsce 4 czerwca 1989 roku z wygraną „Solidarności”, po których nastąpiły pokojowe przemiany w sąsiednich krajach i upadek muru berlińskiego.",
        "followup_q": "Kiedy Polska przystąpiła do NATO i Unii Europejskiej?",
        "followup_a": "Do NATO Polska weszła 12 marca 1999 roku, a do Unii Europejskiej 1 maja 2004 roku.",
    },

    # ── 7. Informatyka i Matematyka (11) ─────────────────────────────────────
    {
        "domain": "informatyka_matematyka",
        "title": "System binarny, bit i bajt",
        "fact": "Komputery reprezentują dane w systemie dwójkowym (bity 0 i 1); jeden bajt liczy 8 bitów i może zapisać 256 różnych wartości (od 0 do 255).",
        "q": "Ile różnych wartości można zapisać w jednym bajcie?",
        "a": "Jeden bajt składa się z 8 bitów, więc daje 2^8 = 256 możliwych kombinacji (np. liczby całkowite od 0 do 255).",
        "followup_q": "Jak kodowanie UTF-8 zapisuje polskie znaki?",
        "followup_a": "Znaki ASCII zajmują w UTF-8 jeden bajt, natomiast polskie litery takie jak ą, ć, ę, ł czy ż zapisywane są za pomocą dwóch bajtów.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Złożoność obliczeniowa i notacja Big-O",
        "fact": "Notacja O(f(n)) opisuje, jak rośnie czas lub pamięć algorytmu wraz ze wzrostem rozmiaru danych wejściowych n (np. O(log n), O(n log n), O(n^2)).",
        "q": "Dlaczego wyszukiwanie binarne O(log n) jest szybsze niż liniowe O(n)?",
        "a": "Wyszukiwanie binarne w posortowanej tablicy w każdym kroku odrzuca połowę pozostałych elementów — dla miliona rekordów potrzebuje najwyżej około 20 porównań zamiast miliona.",
        "followup_q": "Jaką złożoność czasową ma szybkie sortowanie (np. Merge Sort)?",
        "followup_a": "Merge Sort sortuje tablicę w gwarantowanym czasie O(n log n).",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Twierdzenie Pitagorasa",
        "fact": "W każdym trójkącie prostokątnym suma kwadratów długości obu przyprostokątnych równa się kwadratowi długości przeciwprostokątnej: a^2 + b^2 = c^2.",
        "q": "Jak brzmi twierdzenie Pitagorasa?",
        "a": "W trójkącie prostokątnym kwadrat przeciwprostokątnej c jest równy sumie kwadratów przyprostokątnych a i b: a^2 + b^2 = c^2.",
        "followup_q": "Podaj najprostszy przykład trójkąta prostokątnego o całkowitych bokach.",
        "followup_a": "To klasyczna trójka pitagorejska 3, 4 i 5, ponieważ 3^2 + 4^2 = 9 + 16 = 25 = 5^2.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Liczby pierwsze i kryptografia RSA",
        "fact": "Liczba pierwsza ma dokładnie dwa dzielniki naturalne (1 i samą siebie); trudność rozkładu iloczynu ogromnych liczb pierwszych leży u podstaw szyfrowania RSA.",
        "q": "Dlaczego liczby pierwsze są tak ważne w bezpieczeństwie internetu?",
        "a": "Pomnożenie dwóch wielusetcyfrowych liczb pierwszych jest natychmiastowe, ale rozłożenie wyniku z powrotem na czynniki pierwsze zajęłoby współczesnym komputerom miliardy lat — na tym opiera się kryptografia klucza publicznego RSA.",
        "followup_q": "Czy liczba 1 jest liczbą pierwszą?",
        "followup_a": "Nie — liczba pierwsza musi mieć dokładnie dwa różne dzielniki naturalne, a 1 ma tylko jeden dzielnik; najmniejszą liczbą pierwszą jest 2.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Protokół TCP/IP, DNS i HTTP",
        "fact": "W internecie IP adresuje pakiety, TCP zapewnia ich pewny i uporządkowany odbiór, DNS tłumaczy nazwy domen na adresy IP, a HTTP/HTTPS przesyła zasoby WWW.",
        "q": "Co robi serwer DNS, gdy wpisujesz adres strony w przeglądarce?",
        "a": "DNS działa jak książka telefoniczna internetu — zamienia czytelną nazwę domeny na liczbowy adres IP serwera, z którym przeglądarka nawiązuje połączenie.",
        "followup_q": "Co dodaje HTTPS względem zwykłego HTTP?",
        "followup_a": "Szyfrowanie i uwierzytelnianie połączenia za pomocą protokołu TLS, chroniące dane przed podsłuchem i modyfikacją po drodze.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Sieci neuronowe i propagacja wsteczna (backpropagation)",
        "fact": "Sztuczne sieci neuronowe uczą się poprzez obliczanie gradientu funkcji straty względem wag metodą propagacji wstecznej i aktualizację parametrów optymalizatorem np. AdamW.",
        "q": "Na czym polega trening sieci neuronowej z użyciem backpropagation?",
        "a": "Sieć najpierw generuje predykcję w przód, mierzy błąd funkcją straty, a następnie propagacja wsteczna wylicza pochodne (gradienty) pokazujące, w którą stronę i o ile zmienić każdą wagę, by zmniejszyć błąd.",
        "followup_q": "Czym różni się uczenie przez sieć Hopfielda i regułę Hebba?",
        "followup_a": "Plastyczność Hebba i pamięć asocjacyjna Hopfielda modyfikują połączenia lokalnie na podstawie jednoczesnej aktywności sąsiednich neuronów, bez globalnego gradientu wstecznego.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Architektura Transformer i mechanizm uwagi (Attention)",
        "fact": "Mechanizm Self-Attention w architekturze Transformer pozwala każdemu elementowi sekwencji bezpośrednio porównywać się z każdym innym poprzez wektory Query, Key i Value.",
        "q": "Dlaczego mechanizm Attention zrewolucjonizował przetwarzanie języka?",
        "a": "Pozwala modelowi jednocześnie łączyć informacje z odległych miejsc kontekstu (bez sekwencyjnego zapominania jak w starszych RNN) i świetnie zrównolegla się na kartach GPU.",
        "followup_q": "Jak działa Cross-Attention w modelu enkoder-dekoder?",
        "followup_a": "Dekoder generuje zapytania (Query), natomiast klucze (Key) i wartości (Value) pochodzą z zewnętrznej reprezentacji pamięci lub enkodera.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "CPU a GPU — przetwarzanie równoległe",
        "fact": "Procesor CPU ma kilka–kilkadziesiąt bardzo szybkich rdzeni do złożonej logiki sekwencyjnej, a GPU ma tysiące prostych rdzeni i Tensor Cores do masowych obliczeń macierzowych.",
        "q": "Dlaczego sieci neuronowe trenuje się na GPU zamiast na zwykłym CPU?",
        "a": "Trening sieci składa się głównie z miliardów niezależnych mnożeń macierzy, które tysiące rdzeni GPU oraz wyspecjalizowanej jednostki Tensor Cores wykonują równolegle nawet kilkadziesiąt razy szybciej niż CPU.",
        "followup_q": "Po co stosuje się precyzję BF16 lub FP8 na GPU?",
        "followup_a": "Zmniejszają one zużycie pamięci VRAM i dwu- do czterokrotnie przyspieszają obliczenia na Tensor Cores przy zachowaniu wysokiej jakości uczenia.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Pochodna i całka w analizie matematycznej",
        "fact": "Pochodna mierzy chwilowe tempo zmian funkcji (nachylenie stycznej), natomiast całka oznaczona wyznacza pole pod wykresem i sumuje ciągłe przyrosty.",
        "q": "Jaki jest intuicyjny sens pochodnej w fizyce i informatyce?",
        "a": "Jeśli znasz drogę w czasie, jej pierwsza pochodna to chwilowa prędkość, a druga to przyspieszenie; w uczeniu maszynowym pochodne (gradient) wskazują kierunek najszybszego spadku błędu.",
        "followup_q": "Co wiąże pochodną z całką?",
        "followup_a": "Podstawowe twierdzenie rachunku różniczkowego i całkowego mówi, że różniczkowanie i całkowanie są operacjami wzajemnie odwrotnymi.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Bazy danych SQL a NoSQL",
        "fact": "Relacyjne bazy SQL przechowują dane w ustrukturyzowanych tabelach z gwarancjami ACID, podczas gdy bazy NoSQL stawiają na elastyczne schematy i horyzontalne skalowanie.",
        "q": "Co oznacza akronim ACID w bazach danych?",
        "a": "ACID to gwarancje niezawodności transakcji: Atomicity (wszystko albo nic), Consistency (spójność reguł), Isolation (niezależność równoległych operacji) oraz Durability (trwałość zapisu po awarii).",
        "followup_q": "Do czego używa się indeksów w bazie danych?",
        "followup_a": "Indeks (np. B-drzewo) pozwala błyskawicznie znaleźć rekord bez skanowania całej tabeli wiersz po wierszu.",
    },
    {
        "domain": "informatyka_matematyka",
        "title": "Rachunek prawdopodobieństwa i twierdzenie Bayesa",
        "fact": "Twierdzenie Bayesa pozwala aktualizować prawdopodobieństwo hipotezy po uzyskaniu nowego dowodu, łącząc wiedzę wstępną (prior) z wiarygodnością obserwacji.",
        "q": "Do czego służy twierdzenie Bayesa?",
        "a": "Pozwala racjonalnie uaktualniać ocenę prawdopodobieństwa zdarzenia na podstawie nowych faktów — np. obliczyć rzeczywiste ryzyko choroby po pozytywnym wyniku testu o znanej czułości.",
        "followup_q": "Dlaczego ważna jest tzw. częstość bazowa (prior)?",
        "followup_a": "Ponieważ przy bardzo rzadkim zjawisku nawet dobry test może dawać więcej wyników fałszywie dodatnich niż prawdziwych, jeśli zignorujemy niską częstość bazową.",
    },

    # ── 8. Ekonomia i Prawo (10) ─────────────────────────────────────────────
    {
        "domain": "ekonomia_prawo",
        "title": "Popyt, podaż i cena równowagi",
        "fact": "Na wolnym rynku cena ustala się zwykle w punkcie równowagi, gdzie ilość dobra, którą konsumenci chcą kupić (popyt), zrównuje się z ilością oferowaną przez producentów (podaż).",
        "q": "Co dzieje się z ceną produktu, gdy gwałtownie rośnie popyt przy stałej podaży?",
        "a": "Cena zazwyczaj rośnie — wyższa cena zachęca producentów do zwiększenia podaży i ogranicza popyt aż do osiągnięcia nowego punktu równowagi.",
        "followup_q": "Co oznacza pojęcie elastyczności cenowej popytu?",
        "followup_a": "Pokazuje, jak mocno zmienia się sprzedaż po zmianie ceny: leki ratujące życie mają niski popyt elastyczny, a dobra luksusowe lub zamienne — wysoki.",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Inflacja i siła nabywcza pieniądza",
        "fact": "Inflacja to powszechny wzrost ogólnego poziomu cen w gospodarce, oznaczający spadek siły nabywczej pieniądza w czasie.",
        "q": "Czym dokładnie jest inflacja i jak wpływa na oszczędności?",
        "a": "Inflacja oznacza, że za tę samą kwotę pieniędzy można kupić mniej towarów i usług niż wcześniej — jeśli oprocentowanie oszczędności jest niższe od inflacji, ich realna wartość maleje.",
        "followup_q": "Czym jest odwrotność inflacji — deflacja?",
        "followup_a": "Deflacja to trwały spadek ogólnego poziomu cen, który pozornie brzmi korzystnie, ale często hamuje zakupy, inwestycje i zwiększa bezrobocie.",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Procent składany i wartość pieniądza w czasie",
        "fact": "W procencie składanym wypracowane odsetki dolicza się do kapitału, dzięki czemu w kolejnych okresach odsetki naliczane są także od wcześniejszych zysków (wzrost wykładniczy).",
        "q": "Na czym polega mechanizm procentu składanego?",
        "a": "Zysk wypracowany w jednym roku powiększa kapitał bazowy na kolejny rok, co z upływem dekad tworzy efekt kuli śnieżnej zamiast zwykłego wzrostu liniowego.",
        "followup_q": "Jak działa przybliżona reguła 72?",
        "followup_a": "Dzieląc liczbę 72 przez roczną stopę procentową, szybko oszacujesz liczbę lat potrzebną do podwojenia kapitału (np. przy 6% rocznie zajmie to ok. 12 lat).",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "PKB — Produkt Krajowy Brutto",
        "fact": "Produkt Krajowy Brutto (PKB) mierzy łączną wartość rynkową wszystkich dóbr i usług finalnych wytworzonych na terenie danego kraju w określonym czasie.",
        "q": "Co mierzy PKB, a czego nie uwzględnia?",
        "a": "PKB mierzy wielkość produkcji gospodarczej i aktywności rynkowej kraju w ciągu roku lub kwartału, ale nie mierzy bezpośrednio jakości życia, zdrowia, stanu środowiska ani nieodpłatnej pracy w domu.",
        "followup_q": "Dlaczego używa się wskaźnika PKB per capita?",
        "followup_a": "Podzielenie PKB przez liczbę mieszkańców pozwala lepiej porównywać zamożność obywateli między dużymi i małymi państwami.",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Rola banku centralnego i stopy procentowe",
        "fact": "Bank centralny (w Polsce Narodowy Bank Polski) emituje walutę, pilnuje stabilności cen i wpływa na gospodarkę poprzez wysokość bazowych stóp procentowych.",
        "q": "Dlaczego bank centralny podnosi stopy procentowe przy wysokiej inflacji?",
        "a": "Wyższe stopy procentowe podrażają kredyty i uatrakcyjniają oszczędzanie na lokatach, co chłodzi nadmierny popyt w gospodarce i pomaga wyhamować wzrost cen.",
        "followup_q": "Kto w Polsce decyduje o wysokości głównych stóp NBP?",
        "followup_a": "Rada Polityki Pieniężnej (RPP), działająca jako organ decyzyjny Narodowego Banku Polskiego.",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Dywersyfikacja portfela i ryzyko inwestycyjne",
        "fact": "Dywersyfikacja polega na rozłożeniu kapitału pomiędzy nieskorelowane aktywa (np. obligacje, akcje z różnych sektorów, gotówkę), co obniża ryzyko przy podobnej stopie zwrotu.",
        "q": "Dlaczego nie warto lokować całych oszczędności w jednej spółce?",
        "a": "Ponieważ upadek lub kryzys pojedynczej firmy może zniszczyć cały kapitał — rozłożenie środków na wiele niezależnych branż i klas aktywów (dywersyfikacja) ogranicza skutki jednostkowych porażek.",
        "followup_q": "Jakie jest podstawowe prawo zależności między zyskiem a ryzykiem?",
        "followup_a": "Wyższy potencjalny zwrot niemal zawsze wiąże się z większym ryzykiem i wahaniami wartości — obietnica gwarantowanego bardzo wysokiego zysku bez ryzyka to typowy sygnał oszustwa.",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Trójpodział władzy i państwo prawa",
        "fact": "Klasyczny trójpodział władzy Monteskiusza rozdziela władzę ustawodawczą (tworzy prawo), wykonawczą (rządzi) i sądowniczą (rozstrzyga spory niezależnie).",
        "q": "Po co wprowadzono trójpodział władzy?",
        "a": "Aby żadna pojedyncza instytucja nie skupiła pełni kontroli — wzajemne równoważenie i kontrola parlamentu, rządu oraz niezawisłych sądów chronią wolności obywatelskie przed nadużyciami.",
        "followup_q": "Kto w Polsce sprawuje władzę ustawodawczą?",
        "followup_a": "Dwuizbowy parlament złożony z Sejmu (460 posłów) oraz Senatu (100 senatorów).",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Domniemanie niewinności",
        "fact": "Zasada domniemania niewinności mówi, że każdego uważa się za niewinnego, dopóki jego wina nie zostanie udowodniona i stwierdzona prawomocnym wyrokiem sądu.",
        "q": "Kto w procesie karnym musi udowodnić winę według domniemania niewinności?",
        "a": "Ciężar dowodu spoczywa całkowicie na oskarżycielu (prokuraturze) — oskarżony nie musi dowodzić swojej niewinności, a niedające się usunąć wątpliwości rozstrzyga się na jego korzyść.",
        "followup_q": "Czy zasada ta dotyczy też podejrzanego po zatrzymaniu?",
        "followup_a": "Tak — aż do prawomocnego wyroku skazującego osoba ma status podejrzanego lub oskarżonego, nie zaś winnego przestępstwa.",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Koszt alternatywny decyzji",
        "fact": "Koszt alternatywny to wartość najlepszej utraconej możliwości, z której rezygnujemy wybierając inny wariant wykorzystania czasu lub pieniędzy.",
        "q": "Czym jest koszt alternatywny w ekonomii i życiu codziennym?",
        "a": "To realna korzyść z najlepszej alternatywy, którą poświęcasz podejmując dany wybór — na przykład rezygnacja z bezpiecznych odsetek z obligacji, gdy trzymasz pieniądze na nieoprocentowanym koncie.",
        "followup_q": "Czy koszt alternatywny dotyczy tylko pieniędzy?",
        "followup_a": "Nie — dotyczy także czasu i uwagi: godzina przeznaczona na jedno zadanie oznacza godzinę mniej na inne przedsięwzięcie lub odpoczynek.",
    },
    {
        "domain": "ekonomia_prawo",
        "title": "Prawa autorskie i licencje open source",
        "fact": "Prawo autorskie chroni utwory twórcze od chwili ustalenia bez konieczności rejestracji, natomiast licencje open source (np. MIT, GPL) jawnie zezwalają na kopiowanie i modyfikację kodu.",
        "q": "Czym różni się licencja MIT od licencji GPL w oprogramowaniu?",
        "a": "Permisywna licencja MIT pozwala użyć kodu nawet w zamkniętych programach komercyjnych (wystarczy zachować notę o autorze), a licencja GPL (copyleft) wymaga, aby pochodne modyfikacje również były udostępnione jako kod otwarty.",
        "followup_q": "Kiedy utwór przechodzi do domeny publicznej?",
        "followup_a": "W Unii Europejskiej i Polsce majątkowe prawa autorskie wygasają co do zasady 70 lat po śmierci twórcy.",
    },

    # ── 9. Kultura, Psychologia i Nauka (11) ─────────────────────────────────
    {
        "domain": "kultura_nauka",
        "title": "Mikołaj Kopernik i model heliocentryczny",
        "fact": "Mikołaj Kopernik w dziele „O obrotach sfer niebieskich” (1543) wykazał, że Ziemia i inne planety krążą wokół Słońca, zastępując wielowiekowy model geocentryczny.",
        "q": "Co zmieniła teoria Mikołaja Kopernika w postrzeganiu kosmosu?",
        "a": "Kopernik umieścił Słońce zamiast Ziemi w centrum układu planetarnego i wyjaśnił ruchy wsteczne planet na niebie ruchem samej Ziemi wokół Słońca oraz wokół własnej osi.",
        "followup_q": "Jak nazwano przełom naukowy zapoczątkowany przez Kopernika?",
        "followup_a": "Określa się go mianem przewrotu kopernikańskiego, który otworzył nowożytną rewolucję naukową.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Metoda naukowa i falsyfikowalność",
        "fact": "Metoda naukowa łączy obserwację, formułowanie sprawdzalnych hipotez, kontrolowane eksperymenty oraz niezależną weryfikację wyników przez innych badaczy.",
        "q": "Co według Karla Poppera odróżnia teorię naukową od pseudonauki?",
        "a": "Kluczowe jest kryterium falsyfikowalności: prawdziwa teoria naukowa musi przewidywać konkretne fakty i dawać możliwość obalenia jej, gdyby eksperyment wykazał inny wynik.",
        "followup_q": "Dlaczego w nauce powtarza się eksperymenty w różnych laboratoriach?",
        "followup_a": "Replikowalność eliminuje błędy pomiarowe, przypadek i uprzedzenia pojedynczego zespołu, potwierdzając rzetelność odkrycia.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Efekt placebo i podwójnie ślepa próba",
        "fact": "Efekt placebo to realna poprawa samopoczucia pacjenta wywołana samym przekonaniem o przyjmowaniu leczenia, dlatego nowe leki bada się w próbie podwójnie ślepej.",
        "q": "Po co w badaniach klinicznych stosuje się próbę podwójnie ślepą?",
        "a": "Ani pacjent, ani lekarz oceniający objawy nie wiedzą, kto otrzymuje prawdziwy lek, a kto obojętne placebo — eliminuje to sugestię, oczekiwania i stronniczość przy ocenie skuteczności.",
        "followup_q": "Czy efekt placebo działa tylko „w wyobraźni”?",
        "followup_a": "Nie całkiem — oczekiwanie ulgi uruchamia w mózgu realne szlaki endorfin i dopaminy zmniejszające odczuwanie bólu, lecz placebo nie leczy przyczyn chorób zakaźnych ani nowotworów.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Błąd poznawczy potwierdzenia (confirmation bias)",
        "fact": "Efekt potwierdzenia to naturalna skłonność ludzkiego umysłu do wyszukiwania, zapamiętywania i preferowania informacji zgodnych z już posiadanymi przekonaniami.",
        "q": "Jak uniknąć pułapki błędu potwierdzenia w myśleniu?",
        "a": "Warto świadomie szukać najsilniejszych argumentów przeciwko własnej tezie, sprawdzać pierwotne dane i zadawać pytanie: „Co musiałoby się wydarzyć, bym zmienił zdanie?”.",
        "followup_q": "Jaki inny częsty błąd mylenia statystyki z przyczynowością znasz?",
        "followup_a": "Mylenie korelacji z przyczynowością — dwa zjawiska mogą rosnąć razem (np. sprzedaż lodów i liczba utonięć latem) przez wspólny czynnik zewnętrzny, jakim jest upał.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Brzytwa Ockhama",
        "fact": "Brzytwa Ockhama zaleca, aby spośród hipotez równie dobrze wyjaśniających te same fakty wybierać najprostszą, wymagającą najmniej zbędnych założeń.",
        "q": "Jak stosować brzytwę Ockhama w codziennym wnioskowaniu?",
        "a": "Jeśli proste wyjaśnienie bez mnożenia tajemniczych bytów tłumaczy wszystkie znane fakty tak samo dobrze jak skomplikowana teoria spiskowa, najpierw należy przyjąć wersję prostszą.",
        "followup_q": "Czy prostsze wyjaśnienie jest zawsze prawdziwe?",
        "followup_a": "Nie — to reguła metodologiczna, nie absolutny dowód: jeśli pojawią się nowe fakty, których prosta teoria nie wyjaśnia, trzeba ją rozbudować.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Renesans, oświecenie i humanizm",
        "fact": "Renesans (odrodzenie) zwrócił uwagę na człowieka, sztukę antyczną i doświadczenie, a oświecenie postawiło na rozum, edukację, naukę oraz prawa obywatelskie.",
        "q": "Co było głównym hasłem epoki oświecenia?",
        "a": "Oświecenie stawiało na samodzielne używanie rozumu („Sapere aude!” — miej odwagę być mądrym), rozwój nauk przyrodniczych, tolerancję i krytykę ślepych przesądów.",
        "followup_q": "Jakich polskich twórców łączymy z renesansem?",
        "followup_a": "Przede wszystkim Mikołaja Reja, Jana Kochanowskiego oraz w nauce Mikołaja Kopernika i Andrzeja Frycza Modrzewskiego.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Pamięć krótkotrwała i krzywa zapominania Ebbinghausa",
        "fact": "Hermann Ebbinghaus wykazał, że wiedza bez powtórek zanika wykładniczo, a powtórki rozłożone w rosnących odstępach czasu (spaced repetition) najskuteczniej utrwalają ją w pamięci długotrwałej.",
        "q": "Dlaczego powtarzanie materiału przez 4 dni po 15 minut działa lepiej niż jedna długa noc przed egzaminem?",
        "a": "Rozłożone powtórki (spaced repetition) wykorzystują konsolidację śladów podczas snu i wymuszają aktywne odtwarzanie z pamięci zanim ślad całkowicie zaniknie.",
        "followup_q": "Ile elementów naraz mieści zazwyczaj ludzka pamięć robocza?",
        "followup_a": "Współczesne badania wskazują zwykle na około 4 (a z grupowaniem w porcje „chunking” do 7) niezależnych jednostek informacji naraz.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Język polski i rodzina języków słowiańskich",
        "fact": "Język polski należy do grupy języków zachodniosłowiańskich (wraz z czeskim i słowackim), ma 7 przypadków gramatycznych i bogaty system fleksyjny.",
        "q": "Do jakiej rodziny i grupy języków należy język polski?",
        "a": "Należy do rodziny języków indoeuropejskich, a ściślej do podgrupy zachodniosłowiańskiej — jest najbliżej spokrewniony z kaszubskim, czeskim, słowackim i językami łużyckimi.",
        "followup_q": "Ile przypadków ma język polski i dlaczego ułatwia to swobodny szyk wyrazów?",
        "followup_a": "Polszczyzna ma 7 przypadków (mianownik, dopełniacz, celownik, biernik, narzędnik, miejscownik, wołacz) — końcówki wskazują rolę wyrazu w zdaniu, więc szyk słów może być bardzo elastyczny.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Fryderyk Chopin — ambasador polskiej muzyki",
        "fact": "Fryderyk Chopin (1810–1849), urodzony w Żelazowej Woli, tworzył głównie utwory na fortepian, łącząc romantyczną wirtuozerię z polskimi rytmami mazurków i polonezów.",
        "q": "Jakie formy muzyczne rozsławił Fryderyk Chopin na świecie?",
        "a": "Chopin wyniósł na światowe sale koncertowe polonezy i mazurki inspirowane polskim folklorem, a także pisał mistrzowskie nokturny, ballady, etiudy, scherza oraz preludia fortepianowe.",
        "followup_q": "Jakie prestiżowe wydarzenie upamiętnia jego twórczość w Warszawie?",
        "followup_a": "Międzynarodowy Konkurs Pianistyczny im. Fryderyka Chopina, organizowany w Warszawie co pięć lat od 1927 roku.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Logika formalna: modus ponens i modus tollens",
        "fact": "W logice klasycznej ze zdania „Jeśli P, to Q” oraz faktu P wynika Q (modus ponens), natomiast zaprzeczenie Q pozwala bezpiecznie wywnioskować nie-P (modus tollens).",
        "q": "Na czym polega częsty błąd logiczny odwracania implikacji?",
        "a": "Jeśli wiemy, że „kiedy pada deszcz, ulica jest mokra”, a widzimy mokrą ulicę, nie wynika z tego automatycznie, że padał deszcz — mogła ją umyć polewaczka.",
        "followup_q": "Co można jednak wywnioskować pewnie, gdy ulica jest sucha?",
        "followup_a": "Wtedy z reguły modus tollens mamy pewność, że nie padał deszcz.",
    },
    {
        "domain": "kultura_nauka",
        "title": "Kultura starożytnej Grecji i początki demokracji",
        "fact": "Starożytne Ateny wykształciły pierwszą demokrację bezpośrednią oraz podstawy europejskiej filozofii (Sokrates, Platon, Arystoteles), logiki i geometrii.",
        "q": "Czym różniła się demokracja ateńska od współczesnej demokracji?",
        "a": "W Atenach obywatele głosowali osobiście na zgromadzeniu ludowym (demokracja bezpośrednia), ale prawa wyborcze mieli wyłącznie pełnoletni wolni mężczyźni, bez kobiet, cudzoziemców i niewolników.",
        "followup_q": "Kogo nazywamy ojcem logiki formalnej w starożytnej Grecji?",
        "followup_a": "Arystotelesa, który usystematyzował reguły sylogizmów i poprawnego wnioskowania.",
    },

    # ── 10. Technologia i Codzienność (11) ───────────────────────────────────
    {
        "domain": "technologia_codziennosc",
        "title": "Jak działa GPS (nawigacja satelitarna)",
        "fact": "Odbiornik GPS wylicza swoją pozycję mierząc czas przelotu sygnałów radiowych z co najmniej czterech satelitów z poprawką na efekty teorii względności Einsteina.",
        "q": "Dlaczego odbiornik GPS potrzebuje sygnału z minimum 4 satelitów?",
        "a": "Trzy satelity wyznaczają trzy współrzędne przestrzenne (szerokość, długość, wysokość), a czwarty służy do dokładnej synchronizacji zegara odbiornika z atomowymi zegarami satelitów.",
        "followup_q": "Jaką rolę w GPS odgrywa teoria względności Einsteina?",
        "followup_a": "Zegary na satelitach spieszą się przez słabszą grawitację i opóźniają przez szybki ruch na orbicie — bez relatywistycznych korekt błąd pozycji rósłby o ok. 11 km dziennie.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Akumulatory litowo-jonowe",
        "fact": "Akumulator litowo-jonowy magazynuje energię poprzez odwracalny ruch jonów litu między anodą grafitową a katodą z tlenków metali przez ciekły lub polimerowy elektrolit.",
        "q": "Dlaczego nie warto regularnie rozładowywać baterii w telefonie do zera ani przegrzewać jej?",
        "a": "Głębokie rozładowania do 0% oraz wysokie temperatury (powyżej 35–40°C) przyspieszają degradację elektrod i elektrolitu, trwałe obniżając pojemność ogniwa litowo-jonowego.",
        "followup_q": "Jaki zakres naładowania najlepiej wydłuża żywotność ogniw Li-ion?",
        "followup_a": "Utrzymywanie na co dzień poziomu między około 20% a 80% w umiarkowanej temperaturze wyraźnie wydłuża liczbę cykli życia baterii.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Lodówka i pompa ciepła (obieg chłodniczy)",
        "fact": "Lodówka nie „produkuje chłodu”, lecz przepompowuje ciepło z wnętrza na zewnątrz poprzez parowanie czynnika chłodniczego w niskim ciśnieniu i skraplanie w wysokim.",
        "q": "Dlaczego otwarcie drzwi lodówki nie chłodzi pokoju, tylko go ogrzewa?",
        "a": "Ponieważ lodówka oddaje przez tylny skraplacz całe ciepło zabrane ze środka plus dodatkowe ciepło pracy sprężarki elektrycznej, więc bilans energii w pokoju rośnie.",
        "followup_q": "Jak działa domowa pompa ciepła w porównaniu z lodówką?",
        "followup_a": "Na identycznej zasadzie fizycznej, tylko odwróconej kierunkiem — pobiera ciepło z chłodniejszego powietrza lub gruntu na zewnątrz i przekazuje je do ogrzewania domu.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Światłowody i transmisja internetu",
        "fact": "Światłowód przesyła impulsy laserowe wewnątrz cienkiego włókna szklanego dzięki zjawisku całkowitego wewnętrznego odbicia z ogromną przepustowością i minimalnymi stratami.",
        "q": "Dlaczego światłowód jest szybszy i bardziej niezawodny od zwykłego kabla miedzianego?",
        "a": "Impulsy światła nie ulegają zakłóceniom elektromagnetycznym, mają dziesiątki razy mniejsze tłumienie na kilometr i pozwalają jednocześnie przesyłać wiele długości fal (kanałów) w jednym włóknie.",
        "followup_q": "Jak internet łączy kontynenty?",
        "followup_a": "Ponad 95% międzykontynentalnego ruchu internetowego płynie nie przez satelity, lecz przez setki podmorskich kabli światłowodowych leżących na dnie oceanów.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Pasteryzacja i konserwacja żywności",
        "fact": "Pasteryzacja polega na krótkim podgrzaniu żywności zwykle do 70–85°C w celu zniszczenia drobnoustrojów chorobotwórczych przy zachowaniu większości witamin i smaku.",
        "q": "Czym pasteryzacja różni się od sterylizacji UHT?",
        "a": "Klasyczna pasteryzacja używa łagodniejszych temperatur (np. 72°C przez kilkanaście sekund), więc produkt wymaga lodówki i zachowuje świeży smak; UHT podgrzewa mleko do ok. 135°C na 2–4 sekundy, niszcząc też przetrwalniki i dając trwałość bez lodówki.",
        "followup_q": "Skąd pochodzi nazwa „pasteryzacja”?",
        "followup_a": "Od francuskiego mikrobiologa Ludwika Pasteura, który w XIX wieku opracował tę metodę ratowania wina i mleka przed psuciem.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Odnawialne źródła energii i magazyny energii",
        "fact": "Elektrownie wiatrowe i słoneczne nie emitują CO2 podczas pracy, ale ich produkcja zależy od pogody, dlatego sieć potrzebuje magazynów energii i stabilnych źródeł sterowalnych.",
        "q": "Dlaczego energetyka wiatrowa i słoneczna potrzebuje magazynów energii?",
        "a": "Bo produkcja słońca osiąga szczyt w południe, a zapotrzebowanie domów rośnie wieczorem — bez magazynów bateryjnych, elektrowni szczytowo-pompowych lub elektrowni jądrowych trudno bilansować sieć na bieżąco.",
        "followup_q": "Jak działa elektrownia szczytowo-pompowa?",
        "followup_a": "Przy nadmiarze prądu w sieci pompuje wodę do górnego zbiornika, a gdy prądu brakuje, spuszcza wodę przez turbiny jak wielka bateria grawitacyjna.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Półprzewodniki, krzem i tranzystory",
        "fact": "Współczesne procesory mieszczą dziesiątki miliardów mikroskopijnych tranzystorów z domieszkowanego krzemu działających jak błyskawiczne przełączniki 0 i 1.",
        "q": "Do czego w układzie scalonym służy pojedynczy tranzystor?",
        "a": "Tranzystor działa jak sterowany napięciem przełącznik albo wzmacniacz prądu — miliardy połączonych tranzystorów tworzą bramki logiczne wykonujące wszystkie operacje komputera.",
        "followup_q": "Dlaczego do produkcji układów scalonych używa się akurat krzemu?",
        "followup_a": "Krzem jest powszechnie dostępny w piasku, łatwo tworzy czyste kryształy i naturalną warstwę izolatora (dwutlenek krzemu), idealną do fotolitografii.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Kuchenka mikrofalowa i podgrzewanie jedzenia",
        "fact": "Kuchenka mikrofalowa emituje fale o częstotliwości 2,45 GHz, które wprawiają w szybkie drgania polarne cząsteczki wody w żywności, rozgrzewając ją tarciem molekularnym.",
        "q": "Dlaczego mikrofalówka nagrzewa zupę, a suchy talerz pozostaje często chłodniejszy?",
        "a": "Mikrofale najsilniej przekazują energię polarnym cząsteczkom wody, tłuszczu i cukru w potrawie, podczas gdy większość ceramiki i szkła przepuszcza te fale niemal bez pochłaniania.",
        "followup_q": "Czy mikrofalówka czyni jedzenie radioaktywnym?",
        "followup_a": "Nie — mikrofale są promieniowaniem niejonizującym o energii zbyt małej do naruszania jąder atomowych czy wiązań chemicznych; powodują wyłącznie ruch cieplny cząsteczek.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Szczególne zasady pierwszej pomocy (RKO)",
        "fact": "Przy nagłym zatrzymaniu krążenia kluczowe jest natychmiastowe wezwanie pomocy (112/999) i uciskanie środka klatki piersiowej w tempie ok. 100–120 na minutę oraz użycie AED.",
        "q": "Co należy zrobić jako pierwsze, gdy widzisz nieprzytomną osobę, która nie oddycha?",
        "a": "Upewnić się, że miejsce jest bezpieczne, natychmiast zadzwonić pod 112 (lub poprosić konkretną osobę o telefon i przyniesienie AED) oraz rozpocząć rytmiczne uciski środka klatki piersiowej na głębokość 5–6 cm w tempie 100–120/min.",
        "followup_q": "Czym jest urządzenie AED?",
        "followup_a": "Automatyczny defibrylator zewnętrzny — głosowo prowadzi ratownika krok po kroku i sam analizuje rytm serca, podając impuls tylko wtedy, gdy jest to potrzebne.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Bezpieczeństwo haseł i uwierzytelnianie dwuskładnikowe (2FA)",
        "fact": "Bezpieczne konto wymaga długiego, unikalnego hasła dla każdego serwisu (najlepiej z menedżera haseł) oraz drugiego składnika 2FA (aplikacja TOTP lub klucz FIDO2).",
        "q": "Dlaczego nie wolno używać tego samego hasła w kilku serwisach?",
        "a": "Gdy jeden słabiej zabezpieczony portal doświadczy wycieku bazy, automaty atakujące natychmiast testują ten sam e-mail i hasło na bankach, poczcie i portalach społecznościowych.",
        "followup_q": "Co daje włączenie uwierzytelniania dwuskładnikowego (2FA)?",
        "followup_a": "Nawet jeśli przestępca pozna Twoje hasło, nie zaloguje się bez fizycznego dostępu do jednorazowego kodu w aplikacji lub klucza sprzętowego.",
    },
    {
        "domain": "technologia_codziennosc",
        "title": "Zasada działania silnika spalinowego a elektrycznego",
        "fact": "Silnik spalinowy przetwarza energię chemiczną paliwa na ruch z sprawnością ok. 25–42% (reszta to ciepło), natomiast silnik elektryczny osiąga często ponad 90% sprawności.",
        "q": "Skąd bierze się znacznie wyższa sprawność silnika elektrycznego w porównaniu ze spalinowym?",
        "a": "Silnik elektryczny bezpośrednio zamienia pole elektromagnetyczne w moment obrotowy z minimalnymi stratami tarcia i ciepła, a przy hamowaniu może dodatkowo odzyskiwać energię (rekuperacja).",
        "followup_q": "Dlaczego samochód elektryczny przyspiesza bardzo dynamicznie od zera?",
        "followup_a": "Ponieważ silnik elektryczny dostarcza maksymalny moment obrotowy natychmiast od zerowych obrotów bez konieczności redukcji biegów.",
    },
]

# High-value core Polish Wikipedia encyclopedic articles used alongside live "Artykuły na Medal" & "Dobre artykuły"
CORE_WIKI_POLISH_TITLES: list[str] = [
    "Polska", "Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań", "Wisła", "Bałtyk", "Tatry",
    "Ziemia", "Słońce", "Księżyc", "Układ Słoneczny", "Droga Mleczna", "Czarna dziura",
    "Teoria względności", "Mechanika kwantowa", "Grawitacja", "Światło", "Termodynamika",
    "Atom", "Pierwiastek chemiczny", "Woda", "Tlen", "Węgiel", "Azot", "Żelazo",
    "Kwas deoksyrybonukleinowy", "Komórka (biologia)", "Fotosynteza", "Ewolucja biologiczna",
    "Mózg", "Neuron", "Układ odpornościowy", "Szczepionka", "Antybiotyk", "Sen",
    "Komputer", "Internet", "Sztuczna inteligencja", "Sieć neuronowa", "Algorytm",
    "Kryptografia", "Liczba pierwsza", "Rachunek różniczkowy i całkowy", "Prawdopodobieństwo",
    "Ekonomia", "Inflacja", "Produkt krajowy brutto", "Bank centralny", "Prawo", "Konstytucja",
    "Mikołaj Kopernik", "Maria Skłodowska-Curie", "Albert Einstein", "Isaac Newton",
    "Alan Turing", "Fryderyk Chopin", "Adam Mickiewicz", "Józef Piłsudski",
    "Chrzest Polski", "Bitwa pod Grunwaldem", "Konstytucja 3 maja", "Enigma",
    "Renesans", "Oświecenie (epoka)", "Rewolucja przemysłowa", "Unia Europejska",
    "NASA", "Kosmiczny Teleskop Jamesa Webba", "Global Positioning System", "Światłowód",
    "Półprzewodnik", "Nadprzewodnictwo", "Akumulator litowo-jonowy", "Elektrownia jądrowa",
    "Zmiana klimatu", "Efekt cieplarniany", "Tektonika płyt", "Wulkan", "Ocean",
    "Język polski", "Filozofia", "Logika", "Psychologia", "Pamięć człowieka",
    "Zasady dynamiki Newtona", "Układ okresowy pierwiastków", "Bakterie", "Wirusy",
    "Krew", "Serce", "Wątroba", "Nerka", "Mitochondrium", "Białka",
]


def fact_knowledge_path(state_dir: Path | str) -> Path:
    p = Path(state_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "fact_knowledge.jsonl"


def _normalize_fact_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def load_fact_knowledge_records(state_dir: Path | str) -> list[dict[str, Any]]:
    path = fact_knowledge_path(state_dir)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                title = str(rec.get("title", "")).strip()
                fact = str(rec.get("fact", "")).strip()
                q = str(rec.get("q", "")).strip()
                a = str(rec.get("a", "")).strip()
                if not title or not fact or not q or not a:
                    continue
                key = _normalize_fact_title(title)
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                records.append(rec)
    except Exception:
        return []
    return records


def count_fact_knowledge_records(state_dir: Path | str, include_offline: bool = False) -> int:
    saved = len(load_fact_knowledge_records(state_dir))
    return saved + (len(CURATED_FACT_ENCYCLOPEDIA) if include_offline else 0)


def _wiki_api_json(params: dict[str, str], timeout: int = 10) -> dict[str, Any]:
    qs = urllib.parse.urlencode(params)
    url = f"https://pl.wikipedia.org/w/api.php?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def _clean_wiki_extract(text: str) -> list[str]:
    clean = re.sub(r"\([^()]{0,120}\)", " ", text or "")
    clean = re.sub(r"\[\d+\]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) < 28 or is_web_boilerplate(p):
            continue
        if p.lower().startswith(("zobacz też", "przypisy", "linki zewnętrzne", "bibliografia")):
            break
        out.append(p)
        if len(out) >= 3:
            break
    return out


def _is_disambig_or_list(title: str, extract: str) -> bool:
    t_low = (title or "").strip().lower()
    e_low = (extract or "").strip().lower()
    if t_low.startswith(("lista ", "kalendarium ", "kategoria:", "szablon:")):
        return True
    if "(ujednoznacznienie)" in t_low or "może odnosić się do" in e_low or "wieloznaczność" in e_low:
        return True
    if len(e_low) < 140:
        return True
    return False


def _synthesize_wiki_record(title: str, extract: str, full_url: str = "", domain: str = "wikipedia_medalowa") -> dict[str, Any] | None:
    if _is_disambig_or_list(title, extract):
        return None
    sents = _clean_wiki_extract(extract)
    if not sents:
        return None

    fact_parts: list[str] = []
    cur_len = 0
    for s in sents:
        if cur_len + len(s) + 1 > 460 and fact_parts:
            break
        fact_parts.append(s)
        cur_len += len(s) + 1
    fact = " ".join(fact_parts).strip()
    if len(fact) < 120 or is_web_boilerplate(fact):
        return None

    first_ans = " ".join(sents[:2]).strip()[:420]
    if len(first_ans) < 70:
        first_ans = fact[:420]
    clean_t = re.sub(r"\s*\([^)]+\)\s*$", "", title).strip() or title.strip()
    q_txt = f"Czym jest {clean_t}?" if len(clean_t.split()) <= 4 else f"Opowiedz najważniejsze fakty o: {clean_t}."
    followup_q = ""
    followup_a = ""
    if len(sents) >= 3:
        followup_q = f"Co jeszcze warto wiedzieć o temacie {clean_t}?"
        followup_a = sents[2][:360].strip()

    return {
        "source": "wikipedia_curated",
        "domain": domain,
        "title": title.strip(),
        "fact": fact,
        "q": q_txt,
        "a": first_ans,
        "followup_q": followup_q,
        "followup_a": followup_a,
        "url": full_url or f"https://pl.wikipedia.org/wiki/{urllib.parse.quote(title.strip().replace(' ', '_'))}",
    }


def fetch_curated_polish_facts(
    state_dir: Path | str,
    target_count: int = 80,
    stop_event: threading.Event | None = None,
    progress_callback: Any | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """
    Fetch high-quality, peer-reviewed encyclopedic summaries from Polish Wikipedia:
      - Kategoria:Artykuły na Medal (Featured Articles)
      - Kategoria:Dobre artykuły (Good Articles)
      - CORE_WIKI_POLISH_TITLES (essential encyclopedic backbone)
    Deduplicates by normalized title and persists new records to `brain_state/fact_knowledge.jsonl`.
    """
    target_n = max(1, min(300, int(target_count)))
    existing = load_fact_knowledge_records(state_dir)
    seen_titles = {_normalize_fact_title(r.get("title", "")) for r in existing}
    for off in CURATED_FACT_ENCYCLOPEDIA:
        seen_titles.add(_normalize_fact_title(off.get("title", "")))

    candidate_titles: list[tuple[str, str]] = []
    seen_cand: set[str] = set()

    def _add_cand(title: str, domain: str) -> None:
        t = (title or "").strip()
        k = _normalize_fact_title(t)
        if not t or not k or k in seen_titles or k in seen_cand:
            return
        if t.lower().startswith(("lista ", "kategoria:", "wikipedia:", "szablon:")):
            return
        seen_cand.add(k)
        candidate_titles.append((t, domain))

    # 1) Gather peer-reviewed Featured & Good Wikipedia articles via MediaWiki API
    for cat_name, dom_lbl in (
        ("Kategoria:Artykuły na Medal", "wiki_medalowe"),
        ("Kategoria:Dobre artykuły", "wiki_dobre"),
    ):
        if stop_event is not None and stop_event.is_set():
            break
        try:
            data = _wiki_api_json(
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": cat_name,
                    "cmnamespace": "0",
                    "cmlimit": "120",
                    "format": "json",
                },
                timeout=timeout,
            )
            members = data.get("query", {}).get("categorymembers", [])
            if isinstance(members, list):
                for m in members:
                    if isinstance(m, dict) and m.get("title"):
                        _add_cand(str(m["title"]), dom_lbl)
        except Exception:
            pass

    # 2) Mix in core Polish encyclopedia topics
    for core_t in CORE_WIKI_POLISH_TITLES:
        _add_cand(core_t, "wiki_kanon")

    path = fact_knowledge_path(state_dir)
    added = 0
    errors = 0
    new_records: list[dict[str, Any]] = []
    batch_size = 8

    for i in range(0, len(candidate_titles), batch_size):
        if added >= target_n or (stop_event is not None and stop_event.is_set()):
            break
        chunk = candidate_titles[i : i + batch_size]
        domain_by_title = {_normalize_fact_title(t): d for t, d in chunk}
        titles_param = "|".join(t for t, _ in chunk)
        try:
            data = _wiki_api_json(
                {
                    "action": "query",
                    "prop": "extracts|info",
                    "exintro": "1",
                    "explaintext": "1",
                    "redirects": "1",
                    "inprop": "url",
                    "titles": titles_param,
                    "format": "json",
                },
                timeout=timeout,
            )
            pages = data.get("query", {}).get("pages", {})
            if isinstance(pages, dict):
                with path.open("a", encoding="utf-8") as fh:
                    for page in pages.values():
                        if added >= target_n or (stop_event is not None and stop_event.is_set()):
                            break
                        if not isinstance(page, dict) or page.get("missing") is not None:
                            continue
                        p_title = str(page.get("title", "")).strip()
                        p_key = _normalize_fact_title(p_title)
                        if not p_title or p_key in seen_titles:
                            continue
                        p_extract = str(page.get("extract", ""))
                        p_url = str(page.get("fullurl", ""))
                        dom = domain_by_title.get(p_key, "wikipedia_curated")
                        rec = _synthesize_wiki_record(p_title, p_extract, full_url=p_url, domain=dom)
                        if rec is None:
                            continue
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        seen_titles.add(p_key)
                        new_records.append(rec)
                        added += 1
        except Exception:
            errors += 1

        if progress_callback is not None:
            try:
                progress_callback({"done": added, "target": target_n, "errors": errors})
            except Exception:
                pass

    return {
        "added": added,
        "errors": errors,
        "total_saved": len(existing) + added,
        "total_with_offline": len(existing) + added + len(CURATED_FACT_ENCYCLOPEDIA),
        "new_records": new_records,
    }


def build_fact_training_pairs(
    state_dir: Path | str | None = None,
    include_offline: bool = True,
    include_grounded_memory: bool = True,
    include_multiturn: bool = True,
    max_pairs: int = 1400,
) -> list[tuple[str, str]]:
    """
    Build diverse, high-signal factual Q&A pairs for Broca + Wernicke training:
      1) Direct Q -> factual answer
      2) Grounded memory-reading prompt: "[Pamięć] <fact>\\n<question>" -> answer
      3) Multi-turn follow-up: "Kontekst rozmowy: Użytkownik: <q> | Ja: <a> || Odpowiedz na: <followup_q>" -> followup_a
    """
    records: list[dict[str, Any]] = []
    if include_offline:
        records.extend(CURATED_FACT_ENCYCLOPEDIA)
    if state_dir is not None:
        records.extend(load_fact_knowledge_records(state_dir))

    pairs: list[tuple[str, str]] = []
    seen_q: set[str] = set()

    def _add_pair(q: str, a: str) -> None:
        qs = (q or "").strip()
        ans = (a or "").strip()
        if len(qs) < 4 or len(ans) < 12:
            return
        k = qs.lower()
        if k in seen_q:
            return
        seen_q.add(k)
        pairs.append((qs, ans))

    for rec in records:
        q = str(rec.get("q", "")).strip()
        a = str(rec.get("a", "")).strip()
        fact = str(rec.get("fact", "")).strip()
        fq = str(rec.get("followup_q", "")).strip()
        fa = str(rec.get("followup_a", "")).strip()

        _add_pair(q, a)
        if include_grounded_memory and fact and q and a:
            _add_pair(f"[Pamięć] {fact}\n{q}", a)
        if include_multiturn and q and a and fq and fa:
            _add_pair(f"Kontekst rozmowy: Użytkownik: {q} | Ja: {a} || Odpowiedz na: {fq}", fa)
            if include_grounded_memory and fact:
                _add_pair(f"[Pamięć] {fact}\nKontekst rozmowy: Użytkownik: {q} | Ja: {a} || Odpowiedz na: {fq}", fa)

        if len(pairs) >= max(1, int(max_pairs)):
            break

    return pairs[: max(1, int(max_pairs))]


def get_fact_engram_texts(
    state_dir: Path | str | None = None,
    include_offline: bool = True,
    max_items: int = 800,
) -> list[tuple[str, str, str]]:
    """Return `(domain, title, fact_text)` tuples for Hippocampus semantic ingestion."""
    records: list[dict[str, Any]] = []
    if include_offline:
        records.extend(CURATED_FACT_ENCYCLOPEDIA)
    if state_dir is not None:
        records.extend(load_fact_knowledge_records(state_dir))

    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for rec in records:
        title = str(rec.get("title", "")).strip()
        fact = str(rec.get("fact", "")).strip()
        dom = str(rec.get("domain", "wiedza")).strip() or "wiedza"
        k = _normalize_fact_title(title or fact[:60])
        if not fact or k in seen:
            continue
        seen.add(k)
        out.append((dom, title or fact[:60], fact))
        if len(out) >= max(1, int(max_items)):
            break
    return out
