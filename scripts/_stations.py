"""Anchor geography for the Bengaluru study corridor.

Station names and approximate coordinates follow the publicly documented
Namma Metro network (OpenStreetMap / operator line maps). Everything else in
the generated bundle -- road junctions, bus stops, headways, travel-time
observations -- is SYNTHETIC. See SOURCES.md.
"""

# (station_id, display name, lat, lon, [lines])
METRO_STATIONS = [
    ("mg_magadi",      "Magadi Road",              12.9752, 77.5548, ["purple"]),
    ("mg_central",     "Sir M. Visvesvaraya",      12.9740, 77.5806, ["purple"]),
    ("mg_majestic",    "Majestic (Kempegowda)",    12.9757, 77.5729, ["purple", "green"]),
    ("mg_vidhana",     "Vidhana Soudha",           12.9794, 77.5906, ["purple"]),
    ("mg_cubbon",      "Cubbon Park",              12.9782, 77.5960, ["purple"]),
    ("mg_mgroad",      "M.G. Road",                12.9756, 77.6068, ["purple"]),
    ("mg_trinity",     "Trinity",                  12.9731, 77.6169, ["purple"]),
    ("mg_halasuru",    "Halasuru",                 12.9767, 77.6265, ["purple"]),
    ("mg_indiranagar", "Indiranagar",              12.9784, 77.6383, ["purple"]),
    ("mg_svroad",      "Swami Vivekananda Road",   12.9857, 77.6432, ["purple"]),
    ("mg_byappanahalli", "Byappanahalli",          12.9906, 77.6535, ["purple"]),
    ("mg_chickpete",   "Chickpete",                12.9673, 77.5760, ["green"]),
    ("mg_krmarket",    "Krishna Rajendra Market",  12.9600, 77.5760, ["green"]),
    ("mg_natcollege",  "National College",         12.9505, 77.5740, ["green"]),
    ("mg_lalbagh",     "Lalbagh",                  12.9450, 77.5800, ["green"]),
    ("mg_southend",    "South End Circle",         12.9370, 77.5790, ["green"]),
    ("mg_jayanagar",   "Jayanagar",                12.9300, 77.5830, ["green"]),
    ("mg_rvroad",      "Rashtreeya Vidyalaya Road", 12.9215, 77.5800, ["green", "yellow"]),
    ("mg_banashankari", "Banashankari",            12.9150, 77.5730, ["green"]),
    # Yellow Line (RV Road - Bommasandra). The corridor is truncated at
    # Bommanahalli because the study bbox stops there; the real line continues
    # south to Electronic City and Bommasandra.
    ("mg_ragigudda",   "Ragigudda",                12.9142, 77.5905, ["yellow"]),
    ("mg_jayadeva",    "Jayadeva Hospital",        12.9178, 77.5993, ["yellow"]),
    ("mg_btm",         "BTM Layout",               12.9166, 77.6105, ["yellow"]),
    ("mg_silkboard",   "Central Silk Board",       12.9174, 77.6228, ["yellow"]),
    ("mg_bommanahalli", "Bommanahalli",            12.9010, 77.6282, ["yellow"]),
]

# Ordered stopping patterns. Names must exist in METRO_STATIONS.
METRO_LINES = {
    "purple": {
        "name": "Purple Line",
        "colour": "#7B3FA0",
        "stations": [
            "mg_magadi", "mg_majestic", "mg_central", "mg_vidhana", "mg_cubbon",
            "mg_mgroad", "mg_trinity", "mg_halasuru", "mg_indiranagar",
            "mg_svroad", "mg_byappanahalli",
        ],
    },
    "green": {
        "name": "Green Line",
        "colour": "#1E8A4C",
        "stations": [
            "mg_majestic", "mg_chickpete", "mg_krmarket", "mg_natcollege",
            "mg_lalbagh", "mg_southend", "mg_jayanagar", "mg_rvroad",
            "mg_banashankari",
        ],
    },
    "yellow": {
        "name": "Yellow Line",
        "colour": "#D8A400",
        "stations": [
            "mg_rvroad", "mg_ragigudda", "mg_jayadeva", "mg_btm",
            "mg_silkboard", "mg_bommanahalli",
        ],
    },
}

# Bus corridors: ordered lists of (name, lat, lon) waypoints. Stops are
# interpolated along them. Synthetic, but follow plausible arterial alignments.
BUS_CORRIDORS = [
    {
        "route_id": "bus_201", "stop_area": "MG Road corridor",
        "name": "201 Majestic – Indiranagar – Domlur",
        "headway_peak_min": 8, "headway_offpeak_min": 16,
        "waypoints": [
            (12.9757, 77.5729), (12.9760, 77.5900), (12.9755, 77.6070),
            (12.9740, 77.6200), (12.9730, 77.6330), (12.9660, 77.6410),
        ],
    },
    {
        "route_id": "bus_012", "stop_area": "Kanakapura Road",
        "name": "12 Banashankari – Jayanagar – Majestic",
        "headway_peak_min": 6, "headway_offpeak_min": 14,
        "waypoints": [
            (12.9150, 77.5730), (12.9280, 77.5810), (12.9420, 77.5790),
            (12.9580, 77.5760), (12.9700, 77.5745), (12.9757, 77.5729),
        ],
    },
    {
        "route_id": "bus_171", "stop_area": "Koramangala corridor",
        "name": "171 Jayanagar – Koramangala – Domlur",
        "headway_peak_min": 12, "headway_offpeak_min": 24,
        "waypoints": [
            (12.9300, 77.5830), (12.9330, 77.6010), (12.9350, 77.6180),
            (12.9420, 77.6300), (12.9560, 77.6390), (12.9660, 77.6410),
        ],
    },
    {
        "route_id": "bus_500", "stop_area": "Outer Ring Road",
        "name": "500 Ring Road orbital",
        "headway_peak_min": 10, "headway_offpeak_min": 22,
        "waypoints": [
            (12.9215, 77.5800), (12.9290, 77.6100), (12.9400, 77.6350),
            (12.9600, 77.6500), (12.9820, 77.6480), (12.9906, 77.6535),
        ],
    },
    {
        "route_id": "bus_045", "stop_area": "Magadi Road",
        "name": "45 Magadi Road – Chickpete – Lalbagh",
        "headway_peak_min": 14, "headway_offpeak_min": 28,
        "waypoints": [
            (12.9752, 77.5548), (12.9700, 77.5650), (12.9673, 77.5760),
            (12.9560, 77.5790), (12.9450, 77.5800),
        ],
    },
    {
        # Sarjapur Road / Outer Ring Road, the tech-park corridor. This is the
        # only public-transport spine anywhere near Doddakannelli.
        "route_id": "bus_356", "stop_area": "Sarjapur Road",
        "name": "356 Sarjapur Road – Agara – Central Silk Board",
        "headway_peak_min": 9, "headway_offpeak_min": 20,
        "waypoints": [
            (12.9150, 77.6905), (12.9210, 77.6740), (12.9238, 77.6560),
            (12.9232, 77.6440), (12.9175, 77.6395), (12.9174, 77.6228),
        ],
    },
    {
        # 100 Feet Ring Road, Banashankari 3rd Stage. Runs past the PES
        # University campus gate and on to Kathriguppe.
        "route_id": "bus_222", "stop_area": "100 Feet Ring Road",
        "name": "222 Banashankari – Kathriguppe – PES University",
        "headway_peak_min": 11, "headway_offpeak_min": 22,
        "waypoints": [
            (12.9150, 77.5730), (12.9205, 77.5640), (12.9280, 77.5540),
            (12.9330, 77.5430), (12.9346, 77.5353), (12.9420, 77.5310),
        ],
    },
]

# Named places the user can pick in the UI. Off-network destinations are the
# interesting ones -- they are what forces a last-mile ride leg.
PLACES = [
    ("home",         "Home (Vijayanagar)",        12.9722, 77.5498, "residential"),
    ("college",      "College (Shanthinagar)",    12.9612, 77.6042, "education"),
    ("domlur",       "Domlur Office Park",        12.9628, 77.6398, "commercial"),
    ("koramangala",  "Koramangala 5th Block",     12.9352, 77.6245, "commercial"),
    ("hsr_office",   "Office (HSR Layout edge)",  12.9160, 77.6390, "commercial"),
    ("indiranagar_100ft", "Indiranagar 100ft Road", 12.9719, 77.6412, "commercial"),
    ("jayanagar_4b", "Jayanagar 4th Block",       12.9260, 77.5838, "commercial"),
    ("banashankari_home", "Banashankari Home",    12.9163, 77.5702, "residential"),
    ("majestic_bus", "Majestic Bus Station",      12.9776, 77.5715, "transport"),
    ("lalbagh_gate", "Lalbagh West Gate",         12.9490, 77.5830, "leisure"),
    ("mg_road_shops", "M.G. Road",                12.9748, 77.6090, "commercial"),
    ("whitefield_gate", "Old Airport Road Gate",  12.9598, 77.6650, "commercial"),
    ("rv_college",   "R.V. Road Junction",        12.9208, 77.5812, "transport"),
    ("wipro_sarjapur", "Wipro Campus, Doddakannelli (Sarjapur Road)",
     12.9185, 77.6880, "commercial"),
    ("pes_university", "PES University, RR Campus (100 Feet Ring Road)",
     12.9346, 77.5353, "education"),
]
