"""Generate scripts/hosts_all.txt and an SSH config snippet for all DSI machines."""

from pathlib import Path

USER = "vianney.gauthier"
DOMAIN = "polytechnique.fr"

MACHINES = {
    "countries": [
        "allemagne", "angleterre", "autriche", "belgique", "espagne",
        "finlande", "france", "groenland", "hollande", "hongrie",
        "irlande", "islande", "lituanie", "malte", "monaco",
        "pologne", "portugal", "roumanie", "suede",
    ],
    "birds": [
        "albatros", "autruche", "bengali", "coucou", "dindon",
        "epervier", "faisan", "gelinotte", "hibou", "harpie",
        "jabiru", "kamiche", "linotte", "loriol", "mouette",
        "nandou", "ombrette", "perdrix", "quetzal", "quiscale",
        "rouloul", "sitelle", "traquet", "urabu", "verdier",
    ],
    "orchids": [
        "aerides", "barlia", "calanthe", "diuris", "encyclia",
        "epipactis", "gennaria", "habenaria", "isotria", "ipsea",
        "liparis", "lycaste", "malaxis", "neotinea", "oncidium",
        "ophrys", "orchis", "pleione", "pogonia", "serapias",
        "telipogon", "vanda", "vanilla", "xylobium", "zeuxine",
    ],
    "departments": [
        "ain", "allier", "ardennes", "carmor", "charente",
        "cher", "creuse", "dordogne", "doubs", "essonne",
        "finistere", "gironde", "indre", "jura", "landes",
        "loire", "manche", "marne", "mayenne", "morbihan",
        "moselle", "saone", "somme", "vendee", "vosges",
    ],
    "fish": [
        "ablette", "anchois", "anguille", "barbeau", "barbue",
        "baudroie", "brochet", "carrelet", "gardon", "gymnote",
        "labre", "lieu", "lotte", "mulet", "murene",
        "piranha", "raie", "requin", "rouget", "roussette",
        "saumon", "silure", "sole", "thon", "truite",
    ],
    "bones": [
        "acromion", "apophyse", "astragale", "atlas", "axis",
        "coccyx", "cote", "cubitus", "cuboide", "femur",
        "frontal", "humerus", "malleole", "metacarpe", "parietal",
        "perone", "phalange", "radius", "rotule", "sacrum",
        "sternum", "tarse", "temporal", "tibia", "xiphoide",
    ],
    "cars": [
        "bentley", "bugatti", "cadillac", "chrysler", "corvette",
        "ferrari", "fiat", "ford", "jaguar", "lada",
        "maserati", "mazda", "nissan", "niva", "peugeot",
        "pontiac", "porsche", "renault", "rolls", "rover",
        "royce", "simca", "skoda", "venturi", "volvo",
    ],
}

scripts_dir = Path(__file__).parent

# Write hosts_all.txt
hosts_path = scripts_dir / "hosts_all.txt"
ssh_config_path = scripts_dir / "ssh_config_snippet.txt"

with hosts_path.open("w") as hf, ssh_config_path.open("w") as cf:
    for group, names in MACHINES.items():
        hf.write(f"# {group}\n")
        for name in names:
            fqdn = f"{name}.{DOMAIN}"
            hf.write(f"{USER}@{fqdn}\n")
            cf.write(f"Host {name}\n")
            cf.write(f"    User {USER}\n")
            cf.write(f"    HostName {fqdn}\n")
        hf.write("\n")
        cf.write("\n")

total = sum(len(v) for v in MACHINES.values())
print(f"Written {total} hosts to {hosts_path}")
print(f"SSH config snippet written to {ssh_config_path}")
print(f"Append to ~/.ssh/config with: cat {ssh_config_path} >> ~/.ssh/config")
