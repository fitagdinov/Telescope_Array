#define NCOUNTERS 507
#define SDATE 24
#define NDATE 30
#define NALT 2
#define RESP_WIDTH 12
#define FADC_LENGTH 128
#define FADC_PED 6.
#define FADC_NOISE 0.75
#define ONE_MIP_TH 50.
#define COUNTRATE1 500.
#define FADC_RATE 5.e7 
#define NCAL 26
#define FWHMSIG 3.330218
#define TOFFSET 43
#define CSPEED2 29.97925


float NPE2FADCE[2];
float mip[NDATE*144][NCOUNTERS][2];
float mevpoisson[NDATE*144][NCOUNTERS][2];
float one_mev[NDATE*144][NCOUNTERS][2];
float fadc_ped[NDATE*144][NCOUNTERS][2];
float fadc_noise[NDATE*144][NCOUNTERS][2];
float pchped[NDATE*144][NCOUNTERS][2];
float lhpchped[NDATE*144][NCOUNTERS][2];
float rhpchped[NDATE*144][NCOUNTERS][2];
float mftndof[NDATE*144][NCOUNTERS][2];
float mftchi2[NDATE*144][NCOUNTERS][2];
float sat[NDATE*144][NCOUNTERS][2];
char  deadtime[NDATE*144][NCOUNTERS];
int   bsd_bitf[NDATE*144][NCOUNTERS]; // bit flag for a counter that's not working properly

const float ONE_MeV[2] ={19.17, 19.37};
const float one_mip[RESP_WIDTH] = 
  {0.09044301,
   0.2011518,
   0.2357062,
   0.1958141,
   0.1346454,
   0.07871757,
   0.03778135,
   0.01320628,
   0.003927172,
   0.001456995,
   0.0007651797,
   0.0004421589};

const int sdcoor0[NCOUNTERS][4] = {
  {106, -1353650, -1254828, 152800},
  {107, -1348096, -1132747, 151122},
  {108, -1345739, -1014892, 150227},
  {109, -1346994, -893367, 150484},
  {110, -1348677, -770619, 150403},
  {111, -1342444, -652208, 149182},
  {112, -1346457, -532574, 147488},
  {113, -1344006, -409270, 146562},
  {114, -1346042, -292415, 147049},
  {115, -1345999, -167666, 147577},
  {116, -1345799, -51812, 148790},
  {205, -1228013, -1374240, 145700},
  {206, -1221090, -1246267, 147875},
  {207, -1227463, -1133859, 147561},
  {208, -1227446, -1013557, 147615},
  {209, -1227252, -890032, 146782},
  {210, -1227154, -773288, 145925},
  {211, -1226965, -652986, 145710},
  {212, -1226948, -532574, 145853},
  {213, -1226932, -412828, 146418},
  {214, -1226398, -292415, 146127},
  {215, -1226208, -172114, 146990},
  {216, -1226019, -51812, 150302},
  {217, -1219554, 65599, 156838},
  {304, -1111131, -1494430, 143500},
  {305, -1107943, -1373350, 144565},
  {306, -1108118, -1253938, 145457},
  {307, -1107861, -1133637, 145824},
  {308, -1107777, -1013557, 145801},
  {309, -1107607, -893478, 145164},
  {310, -1107436, -773288, 144832},
  {311, -1107265, -652764, 145013},
  {312, -1107526, -532907, 145338},
  {313, -1106924, -412606, 145543},
  {314, -1106753, -292415, 145372},
  {315, -1106583, -172114, 145828},
  {316, -1106412, -51812, 147628},
  {317, -1106241, 68378, 150185},
  {318, -1106070, 188346, 150693},
  {403, -988782, -1614509, 141500},
  {404, -984495, -1496543, 142443},
  {405, -988478, -1374240, 143479},
  {406, -996246, -1247267, 144310},
  {407, -988088, -1133637, 144868},
  {408, -988022, -1013557, 145377},
  {409, -987870, -893367, 144439},
  {410, -987632, -773288, 143903},
  {411, -986704, -652653, 144275},
  {412, -988446, -532018, 144622},
  {413, -986831, -412828, 144954},
  {414, -986851, -292526, 144110},
  {415, -986871, -172114, 144104},
  {416, -987152, -54480, 144601},
  {417, -986566, 68378, 145612},
  {418, -986242, 188124, 146160},
  {419, -983337, 308426, 147010},
  {502, -869413, -1734811, 139600},
  {503, -869280, -1614509, 140879},
  {504, -868974, -1494430, 141909},
  {505, -875047, -1374907, 142244},
  {506, -868535, -1254161, 142681},
  {507, -868401, -1133637, 142494},
  {508, -868267, -1013557, 142114},
  {509, -867787, -891922, 142552},
  {510, -867914, -773288, 143038},
  {511, -868298, -653765, 143689},
  {512, -867302, -532685, 143880},
  {513, -866652, -412606, 143330},
  {514, -867378, -292193, 142861},
  {515, -867073, -172114, 143169},
  {516, -866939, -51812, 143717},
  {517, -866633, 68267, 144508},
  {518, -866671, 188458, 145649},
  {519, -867829, 307091, 146966},
  {520, -866231, 428838, 147684},
  {601, -749840, -1859894, 139200},
  {602, -749634, -1735033, 139680},
  {603, -751151, -1607838, 140313},
  {604, -749318, -1494430, 140587},
  {605, -749202, -1374240, 140983},
  {606, -748915, -1254161, 141504},
  {607, -748627, -1133859, 141785},
  {608, -748426, -1013557, 141377},
  {609, -748397, -893367, 141801},
  {610, -750003, -771954, 142357},
  {611, -748080, -652986, 142734},
  {612, -747965, -532907, 143422},
  {613, -747849, -412272, 142491},
  {614, -749717, -295862, 142182},
  {615, -747275, -172114, 142775},
  {616, -747159, -51812, 143426},
  {617, -746958, 68267, 143646},
  {618, -746843, 187568, 144287},
  {619, -746555, 308426, 145206},
  {620, -746267, 429394, 146503},
  {621, -745293, 548584, 148486},
  {701, -630038, -1855224, 138948},
  {702, -629769, -1735033, 139091},
  {703, -629672, -1614732, 140016},
  {704, -629489, -1494430, 140635},
  {705, -629651, -1374796, 140961},
  {706, -629295, -1254161, 142054},
  {707, -629198, -1133637, 140797},
  {708, -628757, -1013557, 140910},
  {709, -628574, -893478, 141208},
  {710, -628736, -773288, 141425},
  {711, -628381, -653209, 141952},
  {712, -637329, -538244, 142600},
  {713, -628015, -412606, 141838},
  {714, -627746, -292526, 141845},
  {715, -627563, -172114, 142805},
  {716, -627294, -52034, 142730},
  {717, -627197, 68267, 142794},
  {718, -626928, 188458, 143287},
  {719, -626917, 308759, 144104},
  {720, -626734, 428616, 145405},
  {721, -626637, 549251, 147011},
  {722, -626454, 669219, 149120},
  {801, -510326, -1855224, 138566},
  {802, -510248, -1735033, 138783},
  {803, -509997, -1614954, 139971},
  {804, -509746, -1494097, 140496},
  {805, -509582, -1374351, 139819},
  {806, -511661, -1258608, 141091},
  {807, -509253, -1134193, 139992},
  {808, -508916, -1013780, 140148},
  {809, -508924, -893478, 140291},
  {810, -508759, -773288, 140798},
  {811, -508249, -651652, 140799},
  {812, -508430, -532907, 140760},
  {813, -508180, -412606, 141044},
  {814, -508101, -292526, 141836},
  {815, -507937, -172336, 141813},
  {816, -507945, -52257, 142097},
  {817, -507608, 68378, 142088},
  {818, -507357, 188458, 142079},
  {819, -507279, 308759, 143042},
  {820, -506943, 428616, 143666},
  {821, -507122, 549140, 144960},
  {822, -506786, 669219, 147486},
  {823, -504388, 787964, 150951},
  {901, -390529, -1855446, 138100},
  {902, -390382, -1735145, 138620},
  {903, -390323, -1614954, 138748},
  {904, -356295, -1490539, 138811},
  {905, -392447, -1381022, 139523},
  {906, -389798, -1254161, 139378},
  {907, -389652, -1134193, 139447},
  {908, -389161, -1014113, 139487},
  {909, -389618, -893145, 139877},
  {910, -389041, -773399, 140171},
  {911, -388981, -653209, 140025},
  {912, -388577, -533129, 140079},
  {913, -388517, -412828, 140360},
  {914, -388371, -292526, 140543},
  {915, -388139, -172336, 141228},
  {916, -387907, -52257, 141173},
  {917, -387847, 68267, 141069},
  {918, -387787, 188346, 140942},
  {919, -387727, 308426, 141506},
  {920, -387237, 428616, 142219},
  {921, -387349, 549140, 143196},
  {922, -387031, 669219, 145282},
  {923, -386971, 789410, 146917},
  {924, -386482, 909378, 148634},
  {1001, -257444, -1841215, 138501},
  {1002, -270603, -1735145, 138677},
  {1003, -270474, -1610062, 138827},
  {1004, -269658, -1494875, 138583},
  {1005, -270048, -1374796, 138886},
  {1006, -270006, -1254272, 139009},
  {1007, -269878, -1134415, 139175},
  {1008, -269923, -1014113, 139217},
  {1009, -269451, -893701, 139386},
  {1010, -269323, -773621, 139691},
  {1011, -269281, -653320, 139731},
  {1012, -268895, -532574, 139730},
  {1013, -268768, -413050, 139818},
  {1014, -268813, -292749, 139850},
  {1015, -268082, -172114, 140360},
  {1016, -268041, -51701, 140340},
  {1017, -268172, 67823, 140247},
  {1018, -268131, 187902, 140122},
  {1019, -267745, 308426, 140025},
  {1020, -267704, 428505, 140645},
  {1021, -267662, 548695, 142219},
  {1022, -267191, 668997, 143178},
  {1023, -268783, 788743, 144912},
  {1024, -267022, 909378, 146088},
  {1025, -267668, 1029457, 146930},
  {1101, -151106, -1855669, 138647},
  {1102, -169455, -1755047, 138670},
  {1103, -150714, -1615065, 138787},
  {1104, -150691, -1494986, 138710},
  {1105, -150409, -1374796, 138716},
  {1106, -150128, -1254717, 138881},
  {1107, -150191, -1134415, 138998},
  {1108, -149909, -1013780, 139057},
  {1109, -149714, -893478, 139058},
  {1110, -149605, -773844, 139138},
  {1111, -149409, -653320, 139398},
  {1112, -149128, -533129, 139459},
  {1113, -148933, -413050, 139443},
  {1114, -149082, -292749, 139470},
  {1115, -148715, -172669, 139660},
  {1116, -148606, -52590, 139773},
  {1117, -148411, 67823, 139420},
  {1118, -148216, 187902, 139432},
  {1119, -148107, 308203, 139361},
  {1120, -148256, 428727, 139583},
  {1121, -147717, 548695, 140402},
  {1122, -147608, 668774, 141200},
  {1123, -147414, 789076, 141906},
  {1124, -147219, 909378, 143433},
  {1125, -147110, 1029457, 144646},
  {1126, -146658, 1149647, 145835},
  {1201, -31308, -1855780, 138778},
  {1202, -10693, -1748264, 138943},
  {1203, -31040, -1615510, 138877},
  {1204, -30862, -1495209, 138749},
  {1205, -30599, -1374573, 138657},
  {1206, -30422, -1254494, 138804},
  {1207, -30417, -1134637, 138883},
  {1208, -29723, -1014558, 138959},
  {1209, -29977, -894034, 139067},
  {1210, -29887, -773844, 139749},
  {1211, -29537, -653765, 139448},
  {1212, -29705, -533463, 139355},
  {1213, -29442, -413050, 139266},
  {1214, -29007, -292971, 139639},
  {1215, -28831, -173003, 139218},
  {1216, -28826, -52368, 139209},
  {1217, -28822, 67711, 139119},
  {1218, -28645, 187791, 139187},
  {1219, -27953, 307870, 139149},
  {1220, -28121, 428171, 139079},
  {1221, -28116, 548584, 139338},
  {1222, -27768, 668663, 140001},
  {1223, -27678, 788854, 140673},
  {1224, -27502, 909155, 141913},
  {1225, -27325, 1029457, 142939},
  {1226, -27235, 1149536, 144060},
  {1301, 88231, -1856002, 138912},
  {1302, 88477, -1740926, 138857},
  {1303, 88635, -1615621, 138837},
  {1304, 88966, -1495431, 138852},
  {1305, 89039, -1375129, 138841},
  {1306, 92817, -1253938, 138802},
  {1307, 92889, -1132414, 138811},
  {1308, 89429, -1014558, 138952},
  {1309, 89759, -894257, 138881},
  {1310, 89832, -773955, 138937},
  {1311, 90162, -653876, 139080},
  {1312, 90235, -533685, 139088},
  {1313, 90393, -413384, 139108},
  {1314, 90809, -293305, 139028},
  {1315, 90537, -173003, 138998},
  {1316, 90953, -52590, 138973},
  {1317, 91197, 67489, 139051},
  {1318, 91355, 187791, 138973},
  {1319, 91341, 307425, 139044},
  {1320, 91585, 428171, 138929},
  {1321, 91743, 548362, 138731},
  {1322, 91815, 668774, 139068},
  {1323, 92144, 788743, 139630},
  {1324, 92216, 908933, 140505},
  {1325, 92374, 1029012, 141534},
  {1326, 92531, 1149536, 142242},
  {1327, 92861, 1268053, 143000}, 
  {1401, 208203, -1861784, 139229},
  {1402, 208171, -1740704, 139062},
  {1403, 208479, -1603613, 138875},
  {1404, 208618, -1476085, 138911},
  {1405, 206610, -1379910, 138931},
  {1406, 208731, -1255606, 138885},
  {1407, 209130, -1134749, 138900},
  {1408, 211683, -1016782, 138847},
  {1409, 209496, -894479, 138933},
  {1410, 209550, -774400, 139081},
  {1411, 209862, -653876, 138907},
  {1412, 209916, -533797, 138947},
  {1413, 223826, -399930, 138898},
  {1414, 210196, -293527, 138888},
  {1415, 210593, -172892, 138853},
  {1416, 210647, -52924, 138800},
  {1417, 210787, 67267, 138848},
  {1418, 211012, 187457, 138800},
  {1419, 211152, 307647, 138780},
  {1420, 211291, 427949, 138665},
  {1421, 211430, 548139, 138948},
  {1422, 211655, 668330, 138988},
  {1423, 211708, 789632, 139245},
  {1424, 212019, 908822, 139817},
  {1425, 212073, 1028901, 140498},
  {1426, 212383, 1148980, 141928},
  {1427, 212436, 1269393, 143800},
  {1428, 214205, 1385909, 147500}, 
  {1501, 327913, -1856336, 139444},
  {1502, 327949, -1736256, 138976},
  {1503, 315400, -1624961, 138900},
  {1504, 330780, -1497432, 138952},
  {1505, 330298, -1375463, 138945},
  {1506, 328523, -1254828, 138851},
  {1507, 328904, -1135193, 138904},
  {1508, 327819, -1013557, 138867},
  {1509, 332250, -898926, 138953},
  {1510, 329269, -774511, 138812},
  {1511, 329562, -654321, 138892},
  {1512, 329598, -534019, 138845},
  {1513, 325414, -412272, 138798},
  {1514, 329926, -293527, 138787},
  {1515, 330220, -173337, 138900},
  {1516, 330685, -53369, 138847},
  {1517, 330548, 67044, 138799},
  {1518, 330669, 187235, 138800},
  {1519, 303869, 307314, 138700},
  {1520, 330997, 427727, 138889},
  {1521, 331203, 547917, 138950},
  {1522, 331324, 668107, 139122},
  {1523, 331617, 788298, 139399},
  {1524, 331565, 908599, 139767},
  {1525, 331686, 1028901, 140308},
  {1526, 332064, 1148980, 141280},
  {1527, 332184, 1269059, 146100},
  {1528, 332133, 1389472, 146200},
  {1601, 447453, -1856780, 139353},
  {1602, 447815, -1736256, 139117},
  {1603, 447747, -1616400, 138942},
  {1604, 448281, -1495542, 139016},
  {1605, 448212, -1375908, 138786},
  {1606, 448402, -1255606, 138943},
  {1607, 448677, -1135304, 138994},
  {1608, 448695, -1015114, 138956},
  {1609, 448798, -894812, 138946},
  {1610, 434000, -773844, 139112},
  {1611, 447628, -657434, 139107},
  {1612, 449193, -534575, 138944},
  {1613, 449651, -432730, 139000},
  {1614, 449743, -293861, 138884},
  {1615, 449932, -173559, 138700},
  {1616, 450293, -53369, 138787},
  {1617, 450137, 66711, 138826},
  {1618, 450412, 187012, 138903},
  {1619, 450284, 340352, 139000},
  {1620, 450187, 426281, 139112},
  {1621, 450547, 547472, 139204},
  {1622, 451235, 695125, 139097},
  {1623, 451267, 788187, 139334},
  {1624, 451455, 908377, 139608},
  {1625, 451385, 1028456, 140133},
  {1626, 451658, 1148980, 141067},
  {1627, 445834, 1266947, 143300},
  {1628, 426187, 1379132, 145400},
  {1701, 567251, -1856892, 139340},
  {1702, 567423, -1736701, 139149},
  {1703, 567508, -1616622, 139016},
  {1704, 567938, -1496098, 138879},
  {1705, 567851, -1376019, 138928},
  {1706, 568108, -1255828, 138959},
  {1707, 568279, -1135749, 139097},
  {1708, 568450, -1015225, 139007},
  {1709, 568707, -895035, 139097},
  {1710, 567155, -774177, 138944},
  {1711, 573270, -656655, 139009},
  {1712, 569564, -534797, 138899},
  {1713, 589361, -412828, 139100},
  {1714, 569560, -294083, 138855},
  {1715, 569644, -173892, 138900},
  {1716, 570245, -53702, 138886},
  {1717, 569985, 66711, 138934},
  {1718, 570327, 186790, 139068},
  {1719, 570411, 306758, 139045},
  {1720, 570408, 427393, 139097},
  {1721, 570664, 547472, 139077},
  {1722, 570404, 667885, 139097},
  {1723, 571003, 787742, 139265},
  {1724, 569540, 908266, 139713},
  {1725, 571600, 1028123, 140169},
  {1726, 571769, 1148202, 140458},
  {1727, 571680, 1268615, 140800},
  {1728, 571763, 1388916, 141800},
  {1801, 687136, -1857336, 139311},
  {1802, 687116, -1736812, 139158},
  {1803, 687442, -1616733, 139102},
  {1804, 687681, -1496543, 138710},
  {1805, 687662, -1376464, 138909},
  {1806, 687900, -1255940, 138701},
  {1807, 688053, -1135860, 139088},
  {1808, 688119, -1015670, 138995},
  {1809, 688272, -895368, 139037},
  {1810, 688597, -775512, 139107},
  {1811, 688663, -655210, 138995},
  {1812, 668300, -508335, 138900},
  {1813, 689739, -411382, 138926},
  {1814, 689205, -294194, 138947},
  {1815, 689450, -181675, 138900},
  {1816, 689078, -53924, 139045},
  {1817, 689316, 65821, 139165},
  {1818, 689897, 186790, 139094},
  {1819, 690221, 306869, 139114},
  {1820, 690286, 426948, 139175},
  {1821, 681315, 556923, 139097},
  {1822, 689303, 663104, 139137},
  {1823, 689799, 782627, 139097},
  {1824, 690805, 907821, 139370},
  {1825, 691041, 1028123, 139750},
  {1826, 691191, 1148758, 140345},
  {1827, 691085, 1268503, 140100},
  {1828, 691836, 1388694, 140000},
  {1902, 806723, -1737035, 139267},
  {1903, 807031, -1617289, 139177},
  {1904, 807165, -1496543, 138704},
  {1905, 812209, -1372683, 139167},
  {1906, 807693, -1256384, 138937},
  {1907, 807827, -1136416, 139012},
  {1908, 819857, -1022119, 138971},
  {1909, 808181, -895702, 139100},
  {1910, 808574, -775512, 139067},
  {1911, 808449, -654988, 139115},
  {1912, 809186, -535464, 139102},
  {1913, 808803, -414829, 139087},
  {1914, 808935, -294083, 138947},
  {1915, 808811, -174115, 139097},
  {1916, 809289, -54147, 139000},
  {1917, 809594, 65821, 139097},
  {1918, 809641, 186234, 139178},
  {1919, 809774, 306313, 139110},
  {1920, 808965, 422390, 139168},
  {1921, 810211, 546916, 139269},
  {1922, 811035, 663215, 139147},
  {1923, 810562, 787075, 139157},
  {1924, 810608, 907710, 139139},
  {1925, 808163, 1027233, 139244},
  {1926, 810615, 1147979, 139700},
  {1927, 811091, 1268059, 139600},
  {2004, 926995, -1497099, 139126},
  {2005, 927715, -1377353, 139089},
  {2006, 927313, -1256718, 138892},
  {2007, 927429, -1136416, 139039},
  {2008, 927717, -1016226, 139010},
  {2009, 927919, -896147, 138950},
  {2010, 928466, -776623, 138947},
  {2011, 928322, -655988, 139026},
  {2012, 928351, -535464, 139061},
  {2013, 928306, -425614, 138800},
  {2014, 923586, -292415, 139336},
  {2015, 928868, -174560, 139113},
  {2016, 928896, -54036, 139263},
  {2017, 929184, 65599, 139097},
  {2018, 929298, 186234, 139217},
  {2019, 929584, 306313, 139240},
  {2020, 929183, 426392, 139223},
  {2021, 931877, 546138, 139284},
  {2022, 928389, 658879, 139285},
  {2023, 930213, 787075, 139300},
  {2024, 930326, 907265, 139417},
  {2025, 926652, 1033348, 139400},
  {2105, 1047094, -1377131, 138890},
  {2106, 1046934, -1257051, 138862},
  {2107, 1047117, -1136861, 138873},
  {2108, 1047473, -1016559, 139097},
  {2109, 1047656, -896258, 139097},
  {2110, 1047753, -776067, 138938},
  {2111, 1048281, -655988, 139175},
  {2112, 1047428, -534019, 139188},
  {2113, 1048302, -415385, 139183},
  {2114, 1048398, -295306, 138947},
  {2115, 1043328, -173003, 139193},
  {2116, 1048763, -54814, 139093},
  {2117, 1048773, 65599, 139113},
  {2118, 1049299, 185900, 139257},
  {2119, 1049482, 305646, 139106},
  {2120, 1049405, 425836, 139234},
  {2121, 1049239, 540849, 139300},
  {2122, 1051661, 665661, 139397},
  {2123, 1059590, 777179, 139533},
  {2206, 1166468, -1257607, 139763},
  {2207, 1181626, -1136972, 145519},
  {2208, 1175876, -1040824, 147100},
  {2209, 1167394, -896703, 139265},
  {2210, 1167473, -776623, 139069},
  {2211, 1167723, -656099, 139097},
  {2212, 1167801, -535909, 139090},
  {2213, 1168052, -415830, 139243},
  {2214, 1168301, -295195, 139106},
  {2215, 1173285, -175116, 139253},
  {2216, 1168543, -55036, 139197},
  {2217, 1168621, 65154, 139227},
  {2218, 1168871, 185456, 139251},
  {2219, 1169034, 305535, 139247},
  {2220, 1169197, 425725, 139539},
  {2307, 1286752, -1137528, 145983},
  {2308, 1286899, -1017338, 146218},
  {2309, 1275569, -885028, 150665},
  {2310, 1283564, -770619, 139850},
  {2311, 1287424, -656655, 139139},
  {2312, 1310559, -536354, 139610},
  {2313, 1287716, -416163, 139574},
  {2314, 1287956, -300865, 139086},
  {2315, 1288523, -175671, 139097},
  {2316, 1287721, -55147, 139168},
  {2317, 1288297, 65154, 139243},
  {2408, 1406655, -1017671, 145063},
  {2409, 1406869, -897370, 145131},
  {2410, 1377439, -768062, 148684},
  {2411, 1407125, -657100, 145735},
  {2412, 1407339, -536799, 139314}
};

const int altcoor[NALT][5]={{2212, 1154785, -529797, 138900, 1200},
			    {1806, 724955, -1280549, 138800, 1200}};