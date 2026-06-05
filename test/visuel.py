# app.py
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

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Générateur de contenu personnalisé",
    page_icon="🎯",
    layout="wide"
)

# ── Reproductibilité ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# ── Paramètres ───────────────────────────────────────────────────────────────
NB_USERS     = 500
INTERESTS = [
    "fitness", "musique", "technologie", "cuisine", "voyage",
    "cinema", "lecture", "gaming", "mode", "photo",
]

ALPHA = [3.0, 2.8, 2.2, 1.8, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5]
INTEREST_PROBS = stats.dirichlet.rvs(alpha=ALPHA, random_state=42)[0]

ACTIONS = {
    "fitness": [
        "watched workout video", "watched yoga tutorial", "watched HIIT session",
        "bought protein powder", "bought resistance bands", "bought running shoes",
        "liked gym post", "liked transformation photo", "liked fitness reel",
        "joined gym", "completed 30-day challenge", "tracked morning run",
        "read fitness blog", "shared meal prep tip", "followed fitness influencer",
    ],
    "musique": [
        "watched music video", "watched live concert stream", "watched guitar tutorial",
        "bought headphones", "bought concert ticket", "bought vinyl record",
        "liked song", "liked album review", "liked playlist share",
        "attended concert", "streamed album", "created playlist",
        "shared song recommendation", "followed artist", "downloaded podcast episode",
    ],
    "technologie": [
        "watched AI talk", "watched coding tutorial", "watched product review",
        "read tech blog", "read AI research paper", "read dev newsletter",
        "bought laptop", "bought smartwatch", "bought mechanical keyboard",
        "liked AI article", "liked startup post", "liked dev project",
        "contributed to open source", "completed coding challenge",
        "starred GitHub repo", "deployed side project", "followed tech influencer",
    ],
    "cuisine": [
        "watched cooking show", "watched recipe tutorial", "watched baking masterclass",
        "bought ingredients", "bought kitchen gadget", "bought cookbook",
        "liked recipe post", "liked food photo", "liked chef video",
        "tried new recipe", "meal prepped for the week", "visited food market",
        "shared recipe", "followed chef", "reviewed restaurant",
    ],
    "voyage": [
        "watched travel vlog", "watched destination guide", "watched packing tips video",
        "booked flight", "booked hotel", "booked guided tour",
        "bought luggage", "bought travel adapter", "bought travel insurance",
        "liked travel photo", "liked itinerary post",
        "checked into hotel", "reviewed attraction", "shared travel tip",
        "followed travel blogger", "saved destination pin",
    ],
    "cinema": [
        "watched movie", "watched trailer", "watched behind the scenes",
        "watched documentary", "watched film analysis",
        "bought streaming sub", "bought cinema ticket", "bought box set",
        "rated film", "liked movie review", "liked actor post",
        "created watchlist", "shared film recommendation", "followed film critic",
        "attended film festival",
    ],
    "lecture": [
        "read fiction novel", "read sci-fi book", "read biography",
        "read self-help book", "read manga",
        "bought ebook", "bought physical book", "bought audiobook",
        "liked book review", "liked reading list", "liked author quote",
        "joined book club", "shared book recommendation", "followed author",
        "highlighted passage", "finished reading challenge",
    ],
    "gaming": [
        "watched game walkthrough", "watched esports tournament", "watched game review",
        "bought game", "bought gaming headset", "bought controller",
        "liked game clip", "liked streamer highlight", "liked game announcement",
        "played multiplayer session", "completed story mode", "joined gaming community",
        "shared gameplay clip", "followed streamer", "participated in beta test",
    ],
    "mode": [
        "watched fashion show", "watched styling tips video", "watched haul video",
        "bought clothing", "bought sneakers", "bought accessories",
        "liked outfit post", "liked streetwear photo", "liked brand post",
        "followed fashion influencer", "saved outfit inspiration",
        "shared outfit of the day", "visited boutique", "subscribed to brand newsletter",
        "reviewed purchase",
    ],
    "photo": [
        "watched photography tutorial", "watched editing walkthrough", "watched gear review",
        "bought camera lens", "bought tripod", "bought editing software",
        "liked photo", "liked photography tip", "liked landscape shot",
        "edited photo in Lightroom", "uploaded photo portfolio",
        "joined photo community", "shared photography tip", "followed photographer",
        "entered photo contest",
    ],
}

CONTENT_CATALOG = {
    "fitness": [
        "Programme HIIT 30 min", "Yoga débutant — 7 jours", "Plan nutrition semaine",
        "Top 10 exercices cardio", "Guide musculation maison", "Programme running 5K",
        "Stretching post-entraînement", "Recettes smoothies protéinés",
        "Challenge 30 jours abdos", "Guide crossfit débutant",
        "Podcast motivation sportive", "Suivi sommeil & récupération",
        "Programme pilates 4 semaines", "Guide étirements matinaux",
    ],
    "musique": [
        "Playlist Rock 2024", "Top Jazz lo-fi", "Concerts à venir près de chez vous",
        "Histoire du Hip-Hop", "Playlist focus & productivité",
        "Cours guitare en ligne", "Introduction au piano", "Playlist chill soirée",
        "Top albums de l'année", "Guide production musicale",
        "Playlist entraînement haute énergie", "Découvertes indie de la semaine",
        "Cours chant débutant", "Podcast histoire de la musique",
    ],
    "technologie": [
        "Blog IA du MIT", "Podcast No Code", "Cours Python avancé",
        "Les dernières avancées en LLM", "Guide débutant open source",
        "Tutoriel Docker & déploiement", "Introduction au machine learning",
        "Top outils dev 2024", "Guide cybersécurité personnel",
        "Cours JavaScript moderne", "Newsletter tech hebdomadaire",
        "Tutoriel API REST", "Guide AWS débutant",
        "Veille innovation tech", "Atelier data visualization",
    ],
    "cuisine": [
        "Recettes végétariennes rapides", "Cours pâtisserie en ligne",
        "Top ustensiles 2024", "Cuisine du monde en 30 min",
        "Meal prep de la semaine", "Guide épices & assaisonnements",
        "Recettes healthy batch cooking", "Cours sushis maison",
        "Top recettes Instagram cette semaine", "Guide fermentation",
        "Plan alimentaire équilibré", "Cours cuisine méditerranéenne",
        "Recettes smoothie bowl", "Guide vins & accords",
    ],
    "voyage": [
        "Top destinations 2024", "Guide voyage solo", "Astuces bagages cabine",
        "Applications voyage indispensables", "Road trips Europe",
        "Guide voyage budget", "Top plages secrètes", "Itinéraire Asie du Sud-Est",
        "Guide voyage digital nomad", "Top Airbnb insolites",
        "Conseils jet lag", "Guide camping sauvage",
        "Top parcs nationaux", "Voyage culturel au Japon",
    ],
    "cinema": [
        "Top films Netflix ce mois", "Les classiques à voir absolument",
        "Documentaires tendance", "Podcast analyse cinéma",
        "Films primés aux Oscars 2024", "Séries binge-watching du moment",
        "Guide cinéma asiatique", "Top thrillers psychologiques",
        "Sélection festival de Cannes", "Films d'animation à ne pas rater",
        "Cinéma africain émergent", "Top documentaires nature",
        "Guide cinéma indépendant", "Rétrospective Spielberg",
    ],
    "lecture": [
        "Top romans 2024", "Club de lecture en ligne", "Bibliothèque numérique gratuite",
        "Sélection science-fiction incontournable", "Guide speed reading",
        "Top biographies inspirantes", "Sélection développement personnel",
        "Meilleurs mangas du moment", "Podcast littéraire hebdomadaire",
        "Liste livres à lire avant 30 ans", "Guide Goodreads",
        "Top thrillers littéraires", "Sélection poésie contemporaine",
        "Newsletter livres & café",
    ],
    "gaming": [
        "Top jeux 2024", "Guide esports & compétition", "Sélection jeux indé",
        "Tutoriel streaming Twitch", "Top jeux coopératifs",
        "Guide build PC gaming", "Sélection jeux narrative",
        "Top FPS compétitifs", "Découvertes roguelike",
        "Guide gaming mobile", "Sélection jeux stratégie",
        "Podcast gaming hebdomadaire", "Top jeux RPG",
        "Calendrier sorties jeux",
    ],
    "mode": [
        "Tendances mode printemps 2024", "Guide streetwear",
        "Top marques éco-responsables", "Lookbook minimaliste",
        "Guide sneakers 2024", "Astuces dressing capsule",
        "Sélection accessoires tendance", "Guide taille & fit",
        "Top vintage & seconde main", "Inspiration street style Tokyo",
        "Guide colorimétrie", "Podcast mode & culture",
        "Sélection montres abordables", "Lookbook bureau chic",
    ],
    "photo": [
        "Cours photo débutant", "Guide composition & cadrage",
        "Top spots photo dans votre ville", "Tutoriel Lightroom",
        "Guide astrophotographie", "Top appareils photo 2024",
        "Cours retouche portrait", "Inspiration photo de rue",
        "Guide photographie de voyage", "Tutoriel Photoshop",
        "Top comptes photo Instagram", "Guide photo paysage",
        "Atelier photo produit", "Cours photo événementielle",
    ],
}

# ── Fonctions données ─────────────────────────────────────────────────────────
def infer_action_type(action):
    if isinstance(action, str):
        if action.startswith(("watched", "read", "streamed")): return "view"
        if action.startswith("bought") or action == "booked flight": return "buy"
        if action.startswith("liked") or action == "rated film": return "like"
    return "other"

def generate_users(n=NB_USERS):
    profiles = []
    for i in range(n):
        age = int(np.clip(np.random.normal(loc=35, scale=10), 18, 65))
        nb_interests = random.randint(1, 8)
        interests = list(np.random.choice(INTERESTS, size=nb_interests, replace=False, p=INTEREST_PROBS))
        user_actions = {}
        for interest in interests:
            available = ACTIONS[interest]
            k = random.randint(1, max(1, len(available) // 2))
            user_actions[interest] = random.sample(available, k=k)
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
                activity_log.append(random.choice(ACTIONS[category]))
        profiles.append({"name": fake.name(), "age": age, "interests": interests, "activity_log": activity_log})
    return profiles

def profiles_to_dataframes(profiles):
    user_records, log_records = [], []
    base_date = datetime(2024, 1, 1)
    for idx, profile in enumerate(profiles):
        user_records.append({"user_id": idx+1, "name": profile["name"], "age": profile["age"], "interests": profile["interests"]})
        for action in profile["activity_log"]:
            action_type = infer_action_type(action)
            category = next((cat for cat, acts in ACTIONS.items() if action in acts), None)
            delta = timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))
            ts = base_date + delta
            log_records.append({"user_id": idx+1, "action": action, "action_type": action_type, "category": category, "timestamp": ts, "hour": ts.hour})
    return pd.DataFrame(user_records), pd.DataFrame(log_records)

def inject_noise(df):
    df_noisy = df.copy()
    n_dup = int(len(df_noisy) * 0.03)
    df_noisy = pd.concat([df_noisy, df_noisy.sample(n=n_dup, random_state=42)], ignore_index=True)
    n_nan = int(len(df_noisy) * 0.02)
    for idx in df_noisy.sample(n=n_nan, random_state=42).index:
        df_noisy.loc[idx, random.choice(["action", "category", "hour"])] = np.nan
    return df_noisy

def clean_data(users_df, logs_df):
    logs_df = logs_df.drop_duplicates().dropna(subset=["action", "category"]).reset_index(drop=True)
    return users_df, logs_df

# ── Moteur de recommandation ──────────────────────────────────────────────────
class RecommendationEngine:
    def __init__(self, users_df, logs_df):
        self.users_df = users_df
        self.logs_df  = logs_df
        self.catalog  = CONTENT_CATALOG

    def recommend(self, user_id, n=3):
        user = self.users_df[self.users_df["user_id"] == user_id]
        if user.empty: return []
        interests  = user.iloc[0]["interests"]
        user_logs  = self.logs_df[self.logs_df["user_id"] == user_id]
        category_counts = user_logs["category"].dropna().value_counts() if not user_logs.empty else pd.Series()
        sorted_interests = sorted(interests, key=lambda x: category_counts.get(x, 0), reverse=True)
        dominant_type = "view"
        if not user_logs.empty and "action_type" in user_logs.columns:
            type_counts = user_logs["action_type"].dropna().value_counts()
            if not type_counts.empty:
                dominant_type = type_counts.idxmax()
        recommendations = []
        for interest in sorted_interests:
            if interest not in self.catalog: continue
            items = self.catalog[interest].copy()
            if dominant_type == "buy":
                items = [i for i in items if any(w in i.lower() for w in ["guide", "cours", "plan", "top"])] or items
            elif dominant_type == "like":
                items = [i for i in items if any(w in i.lower() for w in ["top", "tendance", "2024"])] or items
            picked = np.random.choice(items, size=min(n, len(items)), replace=False).tolist()
            recommendations.append({"interest": interest, "recommendations": picked, "dominant_type": dominant_type})
        return recommendations

    def build_vector(self, uid, interests):
        all_actions = [a for acts in ACTIONS.values() for a in acts]
        user_logs   = self.logs_df[self.logs_df["user_id"] == uid]
        counts      = user_logs["action"].dropna().value_counts()
        total       = counts.sum() if counts.sum() > 0 else 1
        action_vec  = np.array([counts.get(a, 0) / total for a in all_actions])
        interest_vec = np.array([1.0 if i in interests else 0.0 for i in INTERESTS])
        return np.concatenate([interest_vec, action_vec])

    def find_similar(self, user_id, top_k=3):
        user = self.users_df[self.users_df["user_id"] == user_id]
        if user.empty: return []
        target_vec = self.build_vector(user_id, user.iloc[0]["interests"])
        sims = []
        for _, row in self.users_df.iterrows():
            if row["user_id"] == user_id: continue
            vec = self.build_vector(row["user_id"], row["interests"])
            sim = 1 - cosine(target_vec, vec) if np.any(target_vec) and np.any(vec) else 0.0
            sims.append((row["user_id"], row["name"], sim))
        return sorted(sims, key=lambda x: x[2], reverse=True)[:top_k]

    def recommend_from_similar(self, user_id, n=3):
        user = self.users_df[self.users_df["user_id"] == user_id]
        if user.empty: return []
        similar_users = self.find_similar(user_id, top_k=5)
        if not similar_users: return []
        similar_ids  = [uid for uid, _, _ in similar_users]
        similar_logs = self.logs_df[self.logs_df["user_id"].isin(similar_ids)]
        popular_cats = similar_logs["category"].dropna().value_counts().head(3).index.tolist()
        user_interests = user.iloc[0]["interests"]
        new_cats = [c for c in popular_cats if c not in user_interests]
        results = []
        for cat in new_cats:
            if cat not in self.catalog: continue
            items = np.random.choice(self.catalog[cat], size=min(n, len(self.catalog[cat])), replace=False).tolist()
            results.append({"category": cat, "recommendations": items})
        return results

# ── Initialisation session_state ──────────────────────────────────────────────
if "users_df" not in st.session_state:
    profiles = generate_users()
    users_df, logs_df = profiles_to_dataframes(profiles)
    logs_df = inject_noise(logs_df)
    users_df, logs_df = clean_data(users_df, logs_df)
    st.session_state.users_df = users_df
    st.session_state.logs_df  = logs_df
    st.session_state.engine   = RecommendationEngine(users_df, logs_df)

users_df = st.session_state.users_df
logs_df  = st.session_state.logs_df
engine   = st.session_state.engine

# ════════════════════════════════════════════════════════════════════════════
# INTERFACE
# ════════════════════════════════════════════════════════════════════════════

st.title("🎯 Générateur de contenu personnalisé")
st.caption("Système de recommandation basé sur les profils et comportements utilisateurs")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Sélection utilisateur")
    max_id = int(users_df["user_id"].max())  # ← DYNAMIQUE
    user_id = st.slider("ID utilisateur", 1, max_id, 1)
    user    = users_df[users_df["user_id"] == user_id].iloc[0]
    st.markdown(f"**{user['name']}**")
    st.markdown(f"Âge : {user['age']} ans")
    st.markdown(f"Intérêts : {', '.join(user['interests'])}")
    st.divider()
    page = st.radio("Navigation", ["🏠 Recommandations", "📊 Analyse", "📈 Visualisations", "➕ Ajouter un utilisateur"])

# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — RECOMMANDATIONS
# ════════════════════════════════════════════════════════════════════════
if page == "🏠 Recommandations":
    st.subheader(f"Recommandations pour {user['name']}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Basées sur vos intérêts et activités")
        recs = engine.recommend(user_id)
        for block in recs:
            with st.expander(f"**{block['interest'].upper()}** — activité : {block['dominant_type']}"):
                for item in block["recommendations"]:
                    st.markdown(f"- {item}")

    with col2:
        st.markdown("#### 👥 Découvertes basées sur des profils similaires")
        similar = engine.find_similar(user_id)
        if similar:
            st.markdown("**Utilisateurs similaires :**")
            for uid, name, score in similar:
                st.markdown(f"- {name} *(similarité : {score:.2f})*")
            st.divider()

        collab = engine.recommend_from_similar(user_id)
        if collab:
            st.markdown("**Catégories découvertes :**")
            for block in collab:
                with st.expander(f"**{block['category'].upper()}** — profils similaires"):
                    for item in block["recommendations"]:
                        st.markdown(f"- {item}")
        else:
            st.info("Vos intérêts couvrent déjà les catégories populaires de vos profils similaires. Essayez d'explorer d'autres utilisateurs !")

# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE
# ════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse":
    st.subheader("Analyse statistique")

    logs_clean = logs_df.drop_duplicates().dropna().reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Utilisateurs", len(users_df))
    col2.metric("Logs nettoyés", len(logs_clean))

    st.divider()

    # Chi² intérêts
    st.markdown("#### Test χ² — Distribution des intérêts")
    all_interests  = [i for ints in users_df['interests'] for i in ints]
    interest_counts = pd.Series(all_interests).value_counts()
    observed = [interest_counts.get(i, 0) for i in INTERESTS]
    expected = [sum(observed) * p for p in INTEREST_PROBS]
    expected_int = [int(round(x)) for x in expected]
    expected_int[-1] += sum(observed) - sum(expected_int)
    chi2, p = stats.chisquare(observed, expected_int)
    st.markdown(f"χ² = **{chi2:.4f}** | p = **{p:.6f}**")
    if p < 0.05:
        st.success("Distribution significativement non-uniforme ✅")
    else:
        st.warning("Pas de différence significative")

    st.divider()

    # Chi² corrélation actions
    st.markdown("#### Test χ² — 'watched AI talk' → achat tech")
    user_actions = logs_clean.groupby('user_id')['action'].apply(list).reset_index()
    user_actions['watched_ai']  = user_actions['action'].apply(lambda x: 1 if 'watched AI talk' in x else 0)
    user_actions['bought_tech'] = user_actions['action'].apply(lambda x: 1 if any(a in x for a in ['bought laptop', 'bought smartwatch']) else 0)
    contingency = pd.crosstab(user_actions['watched_ai'], user_actions['bought_tech'])
    st.dataframe(contingency)
    if contingency.shape == (2, 2):
        chi2_c, p_c, _, _ = stats.chi2_contingency(contingency)
        st.markdown(f"χ² = **{chi2_c:.4f}** | p = **{p_c:.4f}**")
        if p_c < 0.05:
            st.success("Corrélation significative ✅")
        else:
            st.warning("Pas de corrélation significative")

# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualisations":
    st.subheader("Visualisations")
    sns.set_theme(style="whitegrid", palette="muted")

    tab1, tab2, tab3 = st.tabs(["Répartition des intérêts", "Heatmap activité", "Recommandations par segment"])

    with tab1:
        interests_exploded = users_df["interests"].explode()
        counts = interests_exploded.value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(counts.index, counts.values, color=sns.color_palette("muted", len(counts)))
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=10)
        ax.set_title("Répartition des intérêts des utilisateurs", fontsize=14, pad=15)
        ax.set_xlabel("Intérêt")
        ax.set_ylabel("Nombre d'utilisateurs")
        ax.set_ylim(0, counts.max() + 50)
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        logs_clean = logs_df.dropna(subset=["category", "hour"]).copy()
        logs_clean["hour"] = logs_clean["hour"].astype(int)
        pivot = logs_clean.pivot_table(index="category", columns="hour", values="action", aggfunc="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white", cbar_kws={"label": "Nombre d'actions"})
        ax.set_title("Intensité d'activité par heure et par catégorie", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        if "age_group" not in users_df.columns:
            users_df["age_group"] = users_df["age"].apply(lambda a: "18-25" if a < 26 else "26-40" if a < 41 else "41+")
            engine.users_df = users_df
            st.session_state.users_df = users_df
        records = []
        for _, u in users_df.iterrows():
            for block in engine.recommend(u["user_id"]):
                records.append({"age_group": u["age_group"], "category": block["interest"]})
        recs_df = pd.DataFrame(records)
        counts  = recs_df.groupby(["age_group", "category"]).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=counts, x="category", y="count", hue="age_group", palette="muted", ax=ax)
        ax.set_title("Catégories les plus recommandées par segment d'âge", fontsize=13)
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════
# PAGE 4 — AJOUTER UN UTILISATEUR
# ════════════════════════════════════════════════════════════════════════
elif page == "➕ Ajouter un utilisateur":
    st.subheader("Ajouter un nouvel utilisateur")

    col1, col2 = st.columns(2)

    with col1:
        new_name = st.text_input("Nom", placeholder="ex. Kouassi Akré")
        new_age  = st.slider("Âge", 18, 65, 25)

    with col2:
        new_interests = st.multiselect(
            "Intérêts",
            options=INTERESTS,
            default=["fitness"]
        )

    available_actions = []
    for interest in new_interests:
        available_actions.extend(ACTIONS.get(interest, []))

    new_actions = st.multiselect(
        "Journal d'activité",
        options=available_actions,
        help="Actions cohérentes avec les intérêts choisis"
    )

    if st.button("✅ Ajouter l'utilisateur"):
        if not new_name:
            st.error("Le nom est obligatoire.")
        elif not new_interests:
            st.error("Sélectionne au moins un intérêt.")
        else:
            new_id   = st.session_state.users_df["user_id"].max() + 1
            new_user = pd.DataFrame([{
                "user_id":   new_id,
                "name":      new_name,
                "age":       new_age,
                "interests": new_interests,
            }])

            base_date = datetime(2024, 1, 1)
            new_logs  = []
            for action in new_actions:
                action_type = infer_action_type(action)
                category    = next((cat for cat, acts in ACTIONS.items() if action in acts), None)
                delta       = timedelta(days=random.randint(0, 180), hours=random.randint(0, 23))
                ts          = base_date + delta
                new_logs.append({
                    "user_id":     new_id,
                    "action":      action,
                    "action_type": action_type,
                    "category":    category,
                    "timestamp":   ts,
                    "hour":        ts.hour,
                })

            # Mettre à jour le session_state
            st.session_state.users_df = pd.concat([st.session_state.users_df, new_user], ignore_index=True)
            if new_logs:
                st.session_state.logs_df = pd.concat([st.session_state.logs_df, pd.DataFrame(new_logs)], ignore_index=True)

            # Mettre à jour le moteur
            st.session_state.engine = RecommendationEngine(st.session_state.users_df, st.session_state.logs_df)

            st.success(f"✅ **{new_name}** ajouté avec l'ID {new_id} !")
            st.info("🔁 Retournez à la page **🏠 Recommandations** pour voir les suggestions pour ce nouvel utilisateur.")