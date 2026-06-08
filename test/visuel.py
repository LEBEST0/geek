# app.py — Générateur de contenu personnalisé basé sur l'IA
# Lancer : streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
from faker import Faker
from functools import reduce, total_ordering
from abc import ABC, abstractmethod

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Générateur de contenu personnalisé", page_icon="🎯", layout="wide")

random.seed(42)
np.random.seed(42)
fake = Faker("fr_FR")
Faker.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
NB_USERS = 500

INTERESTS = [
    "fitness", "musique", "technologie", "cuisine", "voyage",
    "cinéma", "lecture", "gaming", "mode", "photographie",
]

# ── Distribution de Dirichlet (SciPy) ────────────────────────────────────────
ALPHA = [3.0, 2.8, 2.2, 1.8, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5]
INTEREST_PROBS = stats.dirichlet.rvs(alpha=ALPHA, random_state=42)[0]

ACTIVITY_MAPPING = {
    "fitness": [
        "regardé une vidéo d'entraînement", "regardé un tutoriel yoga", "regardé une séance HIIT",
        "acheté des protéines en poudre", "acheté des bandes de résistance", "acheté des chaussures de running",
        "aimé une publication de salle de sport", "aimé une photo de transformation",
        "rejoint une salle de sport", "complété un défi 30 jours",
        "suivi une course matinale", "lu un blog fitness", "partagé un conseil nutrition",
    ],
    "musique": [
        "regardé un clip musical", "regardé un concert en direct", "regardé un tutoriel guitare",
        "acheté des écouteurs", "acheté un billet de concert", "acheté un vinyle",
        "aimé une chanson", "aimé une critique d'album",
        "assisté à un concert", "streamé un album", "créé une playlist",
        "partagé une recommandation musicale", "suivi un artiste",
        "téléchargé un épisode de podcast musical",
    ],
    "technologie": [
        "regardé une conférence sur l'IA", "regardé un tutoriel de programmation", "regardé une revue de produit tech",
        "lu un blog tech", "lu un article de recherche en IA", "lu une newsletter développeur",
        "acheté un ordinateur portable", "acheté une montre connectée", "acheté un clavier mécanique",
        "aimé un article sur l'IA", "aimé une publication startup",
        "contribué à l'open source", "complété un défi de programmation",
        "étoilé un dépôt GitHub", "déployé un projet personnel",
    ],
    "cuisine": [
        "regardé une émission culinaire", "regardé un tutoriel recette", "regardé un cours de pâtisserie",
        "acheté des ingrédients", "acheté un gadget de cuisine", "acheté un livre de recettes",
        "aimé une publication de recette", "aimé une photo culinaire",
        "essayé une nouvelle recette", "préparé les repas de la semaine",
        "visité un marché alimentaire", "partagé une recette", "suivi un chef",
        "laissé un avis sur un restaurant",
    ],
    "voyage": [
        "regardé un vlog de voyage", "regardé un guide de destination", "regardé des conseils de bagages",
        "réservé un vol", "réservé un hôtel", "réservé une visite guidée",
        "acheté une valise", "acheté un adaptateur de voyage", "acheté une assurance voyage",
        "aimé une photo de voyage", "aimé un itinéraire de voyage",
        "enregistré dans un hôtel", "laissé un avis sur une attraction",
        "partagé un conseil de voyage", "suivi un blogueur voyage",
    ],
    "cinéma": [
        "regardé un film", "regardé une bande-annonce", "regardé les coulisses d'un film",
        "regardé un documentaire", "regardé une analyse cinématographique",
        "acheté un abonnement streaming", "acheté un billet de cinéma",
        "noté un film", "aimé une critique de film",
        "créé une liste de films à voir", "partagé une recommandation de film",
        "suivi un critique de cinéma", "assisté à un festival de cinéma",
    ],
    "lecture": [
        "lu un roman", "lu un livre de science-fiction", "lu une biographie",
        "lu un livre de développement personnel", "lu un manga",
        "acheté un ebook", "acheté un livre papier", "acheté un livre audio",
        "aimé une critique de livre", "aimé une liste de lecture",
        "rejoint un club de lecture", "partagé une recommandation de livre",
        "suivi un auteur", "terminé un défi lecture",
    ],
    "gaming": [
        "regardé un walkthrough de jeu", "regardé un tournoi esport", "regardé une revue de jeu",
        "acheté un jeu vidéo", "acheté un casque gaming", "acheté une manette",
        "aimé un clip de jeu", "joué en multijoueur", "complété le mode histoire",
        "rejoint une communauté gaming", "partagé un clip de gameplay",
        "suivi un streamer", "participé à un test bêta",
    ],
    "mode": [
        "regardé un défilé de mode", "regardé des conseils stylisme", "regardé une vidéo haul",
        "acheté des vêtements", "acheté des sneakers", "acheté des accessoires",
        "aimé une publication tenue", "aimé une photo streetwear",
        "suivi un influenceur mode", "sauvegardé une inspiration tenue",
        "partagé une tenue du jour", "visité une boutique", "laissé un avis sur un achat",
    ],
    "photographie": [
        "regardé un tutoriel photo", "regardé un walkthrough de retouche", "regardé une revue de matériel photo",
        "acheté un objectif", "acheté un trépied", "acheté un logiciel de retouche",
        "aimé une photo", "aimé un conseil photo",
        "retouché une photo sur Lightroom", "mis en ligne un portfolio photo",
        "rejoint une communauté photo", "partagé un conseil photo",
        "suivi un photographe", "participé à un concours photo",
    ],
}

CONTENT_CATALOG = {
    "fitness": [
        "Programme HIIT 30 min", "Yoga débutant — 7 jours", "Plan nutrition semaine",
        "Top 10 exercices cardio", "Guide musculation maison", "Programme running 5K",
        "Stretching post-entraînement", "Recettes smoothies protéinés",
        "Challenge 30 jours abdos", "Guide crossfit débutant",
        "Podcast motivation sportive", "Suivi sommeil & récupération",
    ],
    "musique": [
        "Playlist Rock 2024", "Top Jazz lo-fi", "Concerts à venir près de chez vous",
        "Histoire du Hip-Hop", "Playlist focus & productivité",
        "Cours guitare en ligne", "Introduction au piano", "Playlist chill soirée",
        "Top albums de l'année", "Guide production musicale",
        "Playlist entraînement haute énergie", "Découvertes indie de la semaine",
    ],
    "technologie": [
        "Blog IA du MIT", "Podcast No Code", "Cours Python avancé",
        "Les dernières avancées en LLM", "Guide débutant open source",
        "Tutoriel Docker & déploiement", "Introduction au machine learning",
        "Top outils dev 2024", "Guide cybersécurité personnel",
        "Newsletter tech hebdomadaire", "Tutoriel API REST", "Guide AWS débutant",
    ],
    "cuisine": [
        "Recettes végétariennes rapides", "Cours pâtisserie en ligne",
        "Top ustensiles 2024", "Cuisine du monde en 30 min",
        "Meal prep de la semaine", "Guide épices & assaisonnements",
        "Recettes healthy batch cooking", "Cours sushis maison",
        "Top recettes Instagram", "Guide fermentation",
        "Plan alimentaire équilibré", "Cours cuisine méditerranéenne",
    ],
    "voyage": [
        "Top destinations 2024", "Guide voyage solo", "Astuces bagages cabine",
        "Applications voyage indispensables", "Road trips Europe",
        "Guide voyage budget", "Top plages secrètes", "Itinéraire Asie du Sud-Est",
        "Guide voyage digital nomad", "Top Airbnb insolites",
        "Conseils jet lag", "Guide camping sauvage",
    ],
    "cinéma": [
        "Top films Netflix ce mois", "Les classiques à voir absolument",
        "Documentaires tendance", "Podcast analyse cinéma",
        "Films primés aux Oscars 2024", "Séries binge-watching du moment",
        "Guide cinéma asiatique", "Top thrillers psychologiques",
        "Sélection festival de Cannes", "Films d'animation à ne pas rater",
    ],
    "lecture": [
        "Top romans 2024", "Club de lecture en ligne", "Bibliothèque numérique gratuite",
        "Sélection science-fiction incontournable", "Guide speed reading",
        "Top biographies inspirantes", "Sélection développement personnel",
        "Meilleurs mangas du moment", "Podcast littéraire hebdomadaire",
        "Liste livres à lire avant 30 ans", "Top thrillers littéraires",
    ],
    "gaming": [
        "Top jeux 2024", "Guide esports & compétition", "Sélection jeux indé",
        "Tutoriel streaming Twitch", "Top jeux coopératifs",
        "Guide build PC gaming", "Sélection jeux narrative",
        "Top FPS compétitifs", "Découvertes roguelike",
        "Sélection jeux stratégie", "Top jeux RPG",
    ],
    "mode": [
        "Tendances mode printemps 2024", "Guide streetwear",
        "Top marques éco-responsables", "Lookbook minimaliste",
        "Guide sneakers 2024", "Astuces dressing capsule",
        "Sélection accessoires tendance", "Guide taille & fit",
        "Top vintage & seconde main", "Inspiration street style Tokyo",
    ],
    "photographie": [
        "Cours photo débutant", "Guide composition & cadrage",
        "Top spots photo dans votre ville", "Tutoriel Lightroom",
        "Guide astrophotographie", "Top appareils photo 2024",
        "Cours retouche portrait", "Inspiration photo de rue",
        "Guide photographie de voyage", "Top comptes photo Instagram",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE GÉNÉRATION
# ═══════════════════════════════════════════════════════════════════════════════

def infer_action_type(action):
    if isinstance(action, str):
        if action.startswith(("regardé", "lu", "streamé", "téléchargé")): return "view"
        if action.startswith(("acheté", "réservé")): return "buy"
        if action.startswith(("aimé", "noté", "étoilé", "sauvegardé")): return "like"
    return "other"


def generate_users(n=NB_USERS):
    profiles = []
    for i in range(n):
        age = int(np.clip(np.random.normal(loc=35, scale=10), 18, 65))
        nb_interests = random.randint(1, 5)
        interests = list(np.random.choice(INTERESTS, size=nb_interests, replace=False, p=INTEREST_PROBS))

        # Sous-ensemble d'actions par intérêt (pas toutes !)
        user_actions = {}
        for interest in interests:
            available = ACTIVITY_MAPPING[interest]
            k = random.randint(1, max(1, len(available) // 2))
            user_actions[interest] = random.sample(available, k=k)

        # activity_log avec 80% dans ses intérêts, 20% hors intérêts
        activity_log = []
        for _ in range(random.randint(5, 15)):
            if random.random() < 0.80:
                category = random.choice(interests)
            else:
                other = [c for c in INTERESTS if c not in interests]
                category = random.choice(other) if other else random.choice(interests)
            if category in user_actions:
                activity_log.append(random.choice(user_actions[category]))
            else:
                activity_log.append(random.choice(ACTIVITY_MAPPING[category]))

        profiles.append({
            "name": fake.name(),
            "age": age,
            "interests": interests,
            "activity_log": activity_log,
        })
    return profiles


def profiles_to_dataframes(profiles):
    user_records, log_records = [], []
    base_date = datetime(2024, 1, 1)
    for idx, p in enumerate(profiles):
        user_records.append({"user_id": idx+1, "name": p["name"], "age": p["age"], "interests": p["interests"]})
        for action in p["activity_log"]:
            action_type = infer_action_type(action)
            category = next((cat for cat, acts in ACTIVITY_MAPPING.items() if action in acts), None)
            delta = timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))
            ts = base_date + delta
            log_records.append({
                "user_id": idx+1, "action": action, "action_type": action_type,
                "category": category, "timestamp": ts, "hour": ts.hour,
            })
    return pd.DataFrame(user_records), pd.DataFrame(log_records)


def inject_noise(df, dup_fraction=0.03, nan_fraction=0.02):
    df_noisy = df.copy()
    n_dup = int(len(df_noisy) * dup_fraction)
    df_noisy = pd.concat([df_noisy, df_noisy.sample(n=n_dup, random_state=42)], ignore_index=True)
    n_nan = int(len(df_noisy) * nan_fraction)
    for idx in df_noisy.sample(n=n_nan, random_state=42).index:
        df_noisy.loc[idx, random.choice(["action", "category", "hour"])] = np.nan
    return df_noisy


def clean_data(users_df, logs_df):
    logs_df = logs_df.drop_duplicates().dropna(subset=["action", "category"]).reset_index(drop=True)
    return users_df, logs_df


# ═══════════════════════════════════════════════════════════════════════════════
# POO — MOTEUR DE RECOMMANDATION (héritage + polymorphisme)
# ═══════════════════════════════════════════════════════════════════════════════

class ProfilUtilisateur:
    """Encapsulation : attributs privés, getters/setters."""

    def __init__(self, user_id, name, age, interests, activity_log):
        self.__user_id      = user_id
        self.__name         = name
        self.__age          = age
        self.__interests    = interests
        self.__activity_log = activity_log

    @property
    def user_id(self):      return self.__user_id
    @property
    def name(self):         return self.__name
    @property
    def age(self):          return self.__age
    @property
    def interests(self):    return self.__interests
    @property
    def activity_log(self): return self.__activity_log

    def vecteur_interets(self):
        return np.array([1.0 if i in self.__interests else 0.0 for i in INTERESTS])

    def __repr__(self):
        return f"ProfilUtilisateur(name={self.__name}, age={self.__age}, interests={self.__interests})"


class RecommandeurBase(ABC):
    """Classe abstraite — interface commune pour tous les recommandeurs."""

    @abstractmethod
    def recommander(self, profil, n=3):
        """Retourne une liste de dicts {category, recommendations, reason}."""
        pass


class RecommandeurInteret(RecommandeurBase):
    """Recommande basé sur les intérêts déclarés et le type d'action dominant."""

    def __init__(self, catalog, logs_df):
        self.catalog = catalog
        self.logs_df = logs_df

    def recommander(self, profil, n=3):
        user_logs = self.logs_df[self.logs_df["user_id"] == profil.user_id]
        category_counts = user_logs["category"].dropna().value_counts() if not user_logs.empty else pd.Series()
        sorted_interests = sorted(profil.interests, key=lambda x: category_counts.get(x, 0), reverse=True)

        dominant_type = "view"
        if not user_logs.empty and "action_type" in user_logs.columns:
            tc = user_logs["action_type"].dropna().value_counts()
            if not tc.empty:
                dominant_type = tc.idxmax()

        results = []
        for interest in sorted_interests:
            if interest not in self.catalog:
                continue
            items = self.catalog[interest].copy()
            if dominant_type == "buy":
                items = list(filter(lambda i: any(w in i.lower() for w in ["guide", "cours", "plan", "top"]), items)) or items
            elif dominant_type == "like":
                items = list(filter(lambda i: any(w in i.lower() for w in ["top", "tendance", "2024"]), items)) or items
            picked = list(np.random.choice(items, size=min(n, len(items)), replace=False))
            results.append({
                "category": interest,
                "recommendations": picked,
                "reason": f"Basé sur votre intérêt pour '{interest}' (activité dominante : {dominant_type})",
            })
        return results


class RecommandeurSimilarite(RecommandeurBase):
    """Recommande basé sur les profils similaires (filtrage collaboratif)."""

    def __init__(self, catalog, logs_df, profils):
        self.catalog = catalog
        self.logs_df = logs_df
        self.profils = profils

    def _build_vector(self, profil):
        all_actions = [a for acts in ACTIVITY_MAPPING.values() for a in acts]
        user_logs = self.logs_df[self.logs_df["user_id"] == profil.user_id]
        counts = user_logs["action"].dropna().value_counts()
        total = counts.sum() if counts.sum() > 0 else 1
        action_vec = np.array([counts.get(a, 0) / total for a in all_actions])
        interest_vec = profil.vecteur_interets()
        return np.concatenate([interest_vec, action_vec])

    def trouver_similaires(self, profil, top_k=5):
        target_vec = self._build_vector(profil)
        sims = []
        for autre in self.profils:
            if autre.user_id == profil.user_id:
                continue
            vec = self._build_vector(autre)
            sim = 1 - cosine(target_vec, vec) if np.any(target_vec) and np.any(vec) else 0.0
            sims.append((autre, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def recommander(self, profil, n=3):
        similar_profil = self.trouver_similaires(profil, top_k=5)
        if not similar_profil:
            return []
        similar_ids = [p.user_id for p, _ in similar_profil]
        similar_logs = self.logs_df[self.logs_df["user_id"].isin(similar_ids)]
        popular_cats = similar_logs["category"].dropna().value_counts().head(3).index.tolist()
        new_cats = list(filter(lambda c: c not in profil.interests, popular_cats))

        results = []
        for cat in new_cats:
            if cat not in self.catalog:
                continue
            items = list(np.random.choice(self.catalog[cat], size=min(n, len(self.catalog[cat])), replace=False))
            results.append({
                "category": cat,
                "recommendations": items,
                "reason": f"Les utilisateurs avec un profil similaire apprécient '{cat}'",
            })
        return results


class MoteurHybride:
    """Combine les deux recommandeurs — polymorphisme."""

    def __init__(self, recommandeurs):
        self.recommandeurs = recommandeurs

    def recommander(self, profil, n=3):
        all_results = []
        for reco in self.recommandeurs:  # polymorphisme : même interface .recommander()
            all_results.extend(reco.recommander(profil, n))
        return all_results

    def trouver_similaires(self, profil, top_k=3):
        for reco in self.recommandeurs:
            if isinstance(reco, RecommandeurSimilarite):
                return reco.trouver_similaires(profil, top_k)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "initialized" not in st.session_state:
    profiles = generate_users()
    users_df, logs_df = profiles_to_dataframes(profiles)
    logs_df = inject_noise(logs_df)
    users_df, logs_df = clean_data(users_df, logs_df)

    profils_obj = [
        ProfilUtilisateur(row["user_id"], row["name"], row["age"], row["interests"],
                          logs_df[logs_df["user_id"] == row["user_id"]]["action"].tolist())
        for _, row in users_df.iterrows()
    ]

    reco_interet = RecommandeurInteret(CONTENT_CATALOG, logs_df)
    reco_sim     = RecommandeurSimilarite(CONTENT_CATALOG, logs_df, profils_obj)
    moteur       = MoteurHybride([reco_interet, reco_sim])

    st.session_state.users_df  = users_df
    st.session_state.logs_df   = logs_df
    st.session_state.profils   = profils_obj
    st.session_state.moteur    = moteur
    st.session_state.initialized = True

users_df = st.session_state.users_df
logs_df  = st.session_state.logs_df
moteur   = st.session_state.moteur

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🎯 Générateur de contenu personnalisé basé sur l'IA")
st.caption("Python · NumPy · Pandas · SciPy · Faker · Matplotlib · Seaborn — POO · Filtrage collaboratif")

with st.sidebar:
    st.header("👤 Sélection utilisateur")
    max_id = int(users_df["user_id"].max())
    user_id = st.slider("ID utilisateur", 1, max_id, 1)
    user = users_df[users_df["user_id"] == user_id].iloc[0]
    st.markdown(f"**{user['name']}**")
    st.markdown(f"Âge : {user['age']} ans")
    st.markdown(f"Intérêts : {', '.join(user['interests'])}")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Recommandations",
        "📊 Analyse statistique",
        "📈 Visualisations",
        "➕ Ajouter un utilisateur",
    ])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — RECOMMANDATIONS
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Recommandations":
    profil = next((p for p in st.session_state.profils if p.user_id == user_id), None)
    if profil is None:
        st.error("Profil introuvable.")
    else:
        st.subheader(f"Recommandations pour {profil.name}")

        # Recommandations par intérêt
        reco_interet = moteur.recommandeurs[0]
        recs_interet = reco_interet.recommander(profil)

        # Recommandations par similarité
        reco_sim = moteur.recommandeurs[1]
        recs_sim = reco_sim.recommander(profil)
        similaires = reco_sim.trouver_similaires(profil, top_k=3)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Basées sur vos intérêts et activités")
            for block in recs_interet:
                with st.expander(f"**{block['category'].upper()}**"):
                    for item in block["recommendations"]:
                        st.markdown(f"- {item}")
                    st.caption(f"→ {block['reason']}")

        with col2:
            st.markdown("### 👥 Découvertes basées sur des profils similaires")
            if similaires:
                st.markdown("**Utilisateurs similaires :**")
                for p_sim, score in similaires:
                    st.markdown(f"- {p_sim.name} *(similarité cosinus : {score:.2f})*")
                st.divider()
            if recs_sim:
                for block in recs_sim:
                    with st.expander(f"**{block['category'].upper()}** — profils similaires"):
                        for item in block["recommendations"]:
                            st.markdown(f"- {item}")
                        st.caption(f"→ {block['reason']}")
            else:
                st.info("Vos intérêts couvrent déjà les catégories populaires de vos profils similaires.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse statistique":
    st.subheader("Analyse statistique — SciPy")

    logs_clean = logs_df.dropna().reset_index(drop=True)

    col1, col2 = st.columns(2)
    col1.metric("Utilisateurs", len(users_df))
    col2.metric("Logs nettoyés", len(logs_clean))

    st.divider()

    # Distribution Dirichlet
    st.markdown("### Distribution des intérêts (Dirichlet)")
    st.markdown("Les probabilités d'intérêts sont générées via `scipy.stats.dirichlet.rvs()` :")
    probs_df = pd.DataFrame({"Intérêt": INTERESTS, "Alpha": ALPHA, "Probabilité": INTEREST_PROBS})
    st.dataframe(probs_df.style.format({"Probabilité": "{:.4f}"}), use_container_width=True)

    st.divider()

    # Chi² distribution
    st.markdown("### Test χ² — Distribution des intérêts")
    all_interests = [i for ints in users_df["interests"] for i in ints]
    freq = pd.Series(all_interests).value_counts()
    observed = [freq.get(i, 0) for i in INTERESTS]
    expected = [sum(observed) * p for p in INTEREST_PROBS]
    expected_int = [int(round(x)) for x in expected]
    expected_int[-1] += sum(observed) - sum(expected_int)
    chi2, p = stats.chisquare(observed, expected_int)
    col1, col2 = st.columns(2)
    col1.metric("χ²", f"{chi2:.4f}")
    col2.metric("p-value", f"{p:.6f}")
    if p < 0.05:
        st.success("✅ Distribution significativement non-uniforme — certains intérêts sont plus populaires")
    else:
        st.warning("Pas de différence significative")

    st.divider()

    # Chi² corrélation
    st.markdown("### Test χ² — 'regardé une conférence sur l'IA' → achat tech ?")
    user_actions = logs_clean.groupby("user_id")["action"].apply(list).reset_index()
    user_actions["vu_ia"] = user_actions["action"].apply(lambda x: 1 if "regardé une conférence sur l'IA" in x else 0)
    user_actions["achete_tech"] = user_actions["action"].apply(lambda x: 1 if any(a in x for a in ["acheté un ordinateur portable", "acheté une montre connectée"]) else 0)
    contingency = pd.crosstab(user_actions["vu_ia"], user_actions["achete_tech"])
    st.dataframe(contingency)
    if contingency.shape == (2, 2):
        chi2_c, p_c, _, _ = stats.chi2_contingency(contingency)
        col1, col2 = st.columns(2)
        col1.metric("χ²", f"{chi2_c:.4f}")
        col2.metric("p-value", f"{p_c:.4f}")
        if p_c < 0.05:
            st.success("✅ Corrélation significative")
        else:
            st.warning("Pas de corrélation significative entre ces deux actions")

    st.divider()

    # Heures de pointe
    st.markdown("### Périodes de pointe d'activité")
    hourly = logs_clean.groupby("hour").size()
    col1, col2 = st.columns(2)
    col1.metric("Heure de pointe", f"{int(hourly.idxmax())}h")
    col2.metric("Heure creuse", f"{int(hourly.idxmin())}h")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualisations":
    st.subheader("Visualisations — Matplotlib & Seaborn")
    sns.set_theme(style="whitegrid", palette="muted")

    tab1, tab2, tab3 = st.tabs([
        "Répartition des intérêts",
        "Heatmap heure × catégorie",
        "Recommandations par segment",
    ])

    with tab1:
        interests_exploded = users_df["interests"].explode()
        counts = interests_exploded.value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(counts.index, counts.values, color=sns.color_palette("muted", len(counts)))
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(val), ha="center", va="bottom", fontsize=10)
        ax.set_title("Répartition des centres d'intérêt des utilisateurs", fontsize=14, fontweight="bold")
        ax.set_xlabel("Intérêt")
        ax.set_ylabel("Nombre d'utilisateurs")
        ax.set_ylim(0, counts.max() + 40)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        logs_viz = logs_df.dropna(subset=["category", "hour"]).copy()
        logs_viz["hour"] = logs_viz["hour"].astype(int)
        pivot = logs_viz.pivot_table(index="category", columns="hour", values="action", aggfunc="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                    cbar_kws={"label": "Nombre d'actions"})
        ax.set_title("Intensité d'activité par heure et par catégorie", fontsize=14, fontweight="bold")
        ax.set_xlabel("Heure")
        ax.set_ylabel("Catégorie")
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        users_seg = users_df.copy()
        users_seg["age_group"] = users_seg["age"].apply(
            lambda a: "18-25" if a < 26 else "26-40" if a < 41 else "41+"
        )
        reco = moteur.recommandeurs[0]
        records = []
        for _, u in users_seg.iterrows():
            profil = next((p for p in st.session_state.profils if p.user_id == u["user_id"]), None)
            if profil is None:
                continue
            for block in reco.recommander(profil):
                records.append({"age_group": u["age_group"], "category": block["category"]})
        recs_df = pd.DataFrame(records)
        counts_seg = recs_df.groupby(["age_group", "category"]).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=counts_seg, x="category", y="count", hue="age_group", palette="muted", ax=ax)
        ax.set_title("Catégories les plus recommandées par segment d'âge", fontsize=14, fontweight="bold")
        ax.set_xlabel("Catégorie")
        ax.set_ylabel("Nombre de recommandations")
        ax.legend(title="Segment d'âge", loc="best")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AJOUTER UN UTILISATEUR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "➕ Ajouter un utilisateur":
    st.subheader("Ajouter un nouvel utilisateur")

    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Nom", placeholder="ex. Kouassi Akré")
        new_age  = st.slider("Âge", 18, 65, 25)
    with col2:
        new_interests = st.multiselect("Intérêts", options=INTERESTS, default=["fitness"])

    available_actions = list(reduce(lambda acc, i: acc + ACTIVITY_MAPPING.get(i, []), new_interests, []))
    new_actions = st.multiselect("Journal d'activité", options=available_actions,
                                  help="Actions cohérentes avec les intérêts choisis")

    if st.button("✅ Ajouter l'utilisateur"):
        if not new_name:
            st.error("Le nom est obligatoire.")
        elif not new_interests:
            st.error("Sélectionne au moins un intérêt.")
        else:
            new_id = st.session_state.users_df["user_id"].max() + 1

            new_user_row = pd.DataFrame([{
                "user_id": new_id, "name": new_name, "age": new_age, "interests": new_interests,
            }])

            base_date = datetime(2024, 1, 1)
            new_logs = []
            for action in new_actions:
                action_type = infer_action_type(action)
                category = next((cat for cat, acts in ACTIVITY_MAPPING.items() if action in acts), None)
                delta = timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))
                ts = base_date + delta
                new_logs.append({
                    "user_id": new_id, "action": action, "action_type": action_type,
                    "category": category, "timestamp": ts, "hour": ts.hour,
                })

            st.session_state.users_df = pd.concat([st.session_state.users_df, new_user_row], ignore_index=True)
            if new_logs:
                st.session_state.logs_df = pd.concat([st.session_state.logs_df, pd.DataFrame(new_logs)], ignore_index=True)

            new_profil = ProfilUtilisateur(new_id, new_name, new_age, new_interests, new_actions)
            st.session_state.profils.append(new_profil)

            # Reconstruire le moteur
            reco_interet = RecommandeurInteret(CONTENT_CATALOG, st.session_state.logs_df)
            reco_sim = RecommandeurSimilarite(CONTENT_CATALOG, st.session_state.logs_df, st.session_state.profils)
            st.session_state.moteur = MoteurHybride([reco_interet, reco_sim])
            moteur = st.session_state.moteur

            st.success(f"✅ **{new_name}** ajouté avec l'ID {new_id} !")
            st.info("🔁 Allez sur **🏠 Recommandations** pour voir les suggestions.")