import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
from functools import reduce
from faker import Faker
import random

st.set_page_config(page_title="Générateur de contenu IA", page_icon="🎯", layout="wide")

# ── Reproductibilité ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)
Faker.seed(42)
fake = Faker('fr_FR')

# ── Paramètres ────────────────────────────────────────────────────────────────
INTERESTS = [
    "fitness", "musique", "technologie", "cuisine", "voyage",
    "cinéma", "lecture", "gaming", "mode", "photographie"
]
ALPHA = [3.0, 2.8, 2.2, 1.8, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5]
INTEREST_PROBS = stats.dirichlet.rvs(alpha=ALPHA, random_state=42)[0]

ACTIVITY_MAPPING = {
    "fitness":      ["regardé une vidéo d'entraînement","regardé un tutoriel yoga","regardé une séance HIIT","acheté des protéines en poudre","acheté des bandes de résistance","acheté des chaussures de running","aimé une publication de salle de sport","aimé une photo de transformation","rejoint une salle de sport","complété un défi 30 jours","suivi une course matinale","lu un blog fitness","partagé un conseil nutrition"],
    "musique":      ["regardé un clip musical","regardé un concert en direct","regardé un tutoriel guitare","acheté des écouteurs","acheté un billet de concert","acheté un vinyle","aimé une chanson","aimé une critique d'album","assisté à un concert","streamé un album","créé une playlist","partagé une recommandation musicale","suivi un artiste","téléchargé un épisode de podcast musical"],
    "technologie":  ["regardé une conférence sur l'IA","regardé un tutoriel de programmation","regardé une revue de produit tech","lu un blog tech","lu un article de recherche en IA","lu une newsletter développeur","acheté un ordinateur portable","acheté une montre connectée","acheté un clavier mécanique","aimé un article sur l'IA","aimé une publication startup","contribué à l'open source","complété un défi de programmation","étoilé un dépôt GitHub","déployé un projet personnel"],
    "cuisine":      ["regardé une émission culinaire","regardé un tutoriel recette","regardé un cours de pâtisserie","acheté des ingrédients","acheté un gadget de cuisine","acheté un livre de recettes","aimé une publication de recette","aimé une photo culinaire","essayé une nouvelle recette","préparé les repas de la semaine","visité un marché alimentaire","partagé une recette","suivi un chef","laissé un avis sur un restaurant"],
    "voyage":       ["regardé un vlog de voyage","regardé un guide de destination","regardé des conseils de bagages","réservé un vol","réservé un hôtel","réservé une visite guidée","acheté une valise","acheté un adaptateur de voyage","acheté une assurance voyage","aimé une photo de voyage","aimé un itinéraire de voyage","enregistré dans un hôtel","laissé un avis sur une attraction","partagé un conseil de voyage","suivi un blogueur voyage"],
    "cinéma":       ["regardé un film","regardé une bande-annonce","regardé les coulisses d'un film","regardé un documentaire","regardé une analyse cinématographique","acheté un abonnement streaming","acheté un billet de cinéma","noté un film","aimé une critique de film","créé une liste de films à voir","partagé une recommandation de film","suivi un critique de cinéma","assisté à un festival de cinéma"],
    "lecture":      ["lu un roman","lu un livre de science-fiction","lu une biographie","lu un livre de développement personnel","lu un manga","acheté un ebook","acheté un livre papier","acheté un livre audio","aimé une critique de livre","aimé une liste de lecture","rejoint un club de lecture","partagé une recommandation de livre","suivi un auteur","terminé un défi lecture"],
    "gaming":       ["regardé un walkthrough de jeu","regardé un tournoi esport","regardé une revue de jeu","acheté un jeu vidéo","acheté un casque gaming","acheté une manette","aimé un clip de jeu","joué en multijoueur","complété le mode histoire","rejoint une communauté gaming","partagé un clip de gameplay","suivi un streamer","participé à un test bêta"],
    "mode":         ["regardé un défilé de mode","regardé des conseils stylisme","regardé une vidéo haul","acheté des vêtements","acheté des sneakers","acheté des accessoires","aimé une publication tenue","aimé une photo streetwear","suivi un influenceur mode","sauvegardé une inspiration tenue","partagé une tenue du jour","visité une boutique","laissé un avis sur un achat"],
    "photographie": ["regardé un tutoriel photo","regardé un walkthrough de retouche","regardé une revue de matériel photo","acheté un objectif","acheté un trépied","acheté un logiciel de retouche","aimé une photo","aimé un conseil photo","retouché une photo sur Lightroom","mis en ligne un portfolio photo","rejoint une communauté photo","partagé un conseil photo","suivi un photographe","participé à un concours photo"],
}

RECOMMANDATIONS_PAR_INTERET = {
    "fitness":      ["Programme HIIT 30 min","Plan nutrition semaine","Guide musculation maison","Défi 30 jours abdos","Programme running 5 km","Podcast motivation sportive"],
    "musique":      ["Playlist Rock 2024","Top Jazz lo-fi","Cours guitare en ligne","Podcast histoire de la musique","Top albums de l'année","Découvertes indie de la semaine"],
    "technologie":  ["Blog IA du MIT","Cours Python avancé","Guide débutant open source","Newsletter tech hebdomadaire","Introduction au machine learning","Tutoriel Docker & déploiement"],
    "cuisine":      ["Recettes végétariennes rapides","Meal prep de la semaine","Cours pâtisserie en ligne","Guide épices et assaisonnements","Cuisine du monde en 30 min","Guide fermentation maison"],
    "voyage":       ["Top destinations 2024","Guide voyage en solo","Astuces bagages cabine","Road trips en Europe","Guide voyage budget","Itinéraire Asie du Sud-Est"],
    "cinéma":       ["Top films Netflix ce mois","Documentaires tendance","Top thrillers psychologiques","Rétrospective Spielberg","Guide cinéma indépendant","Sélection festival de Cannes"],
    "lecture":      ["Top romans 2024","Sélection science-fiction","Top biographies inspirantes","Podcast littéraire hebdomadaire","Guide speed reading","Newsletter livres & café"],
    "gaming":       ["Top jeux 2024","Sélection jeux indépendants","Top jeux coopératifs","Guide configuration PC gaming","Podcast gaming hebdomadaire","Calendrier des sorties jeux"],
    "mode":         ["Tendances mode printemps 2024","Guide sneakers 2024","Lookbook minimaliste","Top vintage et seconde main","Guide colorimétrie","Astuces dressing capsule"],
    "photographie": ["Cours photo débutant","Tutoriel Lightroom","Guide composition et cadrage","Top appareils photo 2024","Guide astrophotographie","Inspiration photo de rue"],
}

# ── Fonctions utilitaires : conversion chaîne ↔ liste ────────────────────────
def liste_vers_chaine(lst):
    """Convertit une liste ['a', 'b', 'c'] en chaîne 'a,b,c'."""
    return ",".join(lst)

def chaine_vers_liste(s):
    """Convertit une chaîne 'a,b,c' en liste ['a', 'b', 'c'].
    Retourne une liste vide si la valeur est absente."""
    if not isinstance(s, str) or s.strip() == "":
        return []
    return [elt.strip() for elt in s.split(",")]

# ── Classes POO ───────────────────────────────────────────────────────────────
class ProfilUtilisateur:
    """Les attributs interests et activity_log sont stockés en interne
    sous forme de chaîne 'elt1,elt2,...' et exposés en liste via propriétés."""

    def __init__(self, name, age, interests, activity_log):
        self.__name         = name
        self.__age          = age
        # Accepte liste ou chaîne en entrée, stocke toujours en chaîne
        self.__interests    = liste_vers_chaine(interests)    if isinstance(interests, list)    else interests
        self.__activity_log = liste_vers_chaine(activity_log) if isinstance(activity_log, list) else activity_log

    @property
    def name(self):         return self.__name
    @property
    def age(self):          return self.__age
    @property
    def interests(self):    return chaine_vers_liste(self.__interests)
    @property
    def activity_log(self): return chaine_vers_liste(self.__activity_log)

    def vecteur_interets(self):
        v_interests = np.array([1.0 if i in self.interests else 0.0 for i in INTERESTS])
        v_actions = np.zeros(len(INTERESTS))
        for idx, interet in enumerate(INTERESTS):
            actions_interet = ACTIVITY_MAPPING.get(interet, [])
            v_actions[idx] = sum(1 for a in self.activity_log if a in actions_interet)
        return v_interests + v_actions

    def description(self):
        return f"Utilisateur standard : {self.__name}, {self.__age} ans"


class ProfilPremium(ProfilUtilisateur):
    def __init__(self, name, age, interests, activity_log, niveau='gold'):
        super().__init__(name, age, interests, activity_log)
        self.niveau = niveau

    def description(self):
        return f"Utilisateur premium ({self.niveau}) : {self.name}, {self.age} ans"


class MoteurRecommandation:
    def __init__(self, df):
        self.profils = [
            ProfilUtilisateur(
                row['name'], row['age'],
                row['interests'],    # chaîne CSV dans le DataFrame
                row['activity_log']  # chaîne CSV dans le DataFrame
            )
            for _, row in df.iterrows()
        ]
        # Calcul du top 10 des contenus les plus populaires (basé sur les intérêts du dataset)
        self.top10_populaires = self._calculer_top10(df)

    def _calculer_top10(self, df):
        """Calcule les 10 contenus les plus populaires en comptant
        combien d'utilisateurs ont chaque intérêt, puis en prenant
        les premières recommandations des intérêts les plus fréquents."""
        tous_interets = [i for s in df['interests'] for i in chaine_vers_liste(str(s)) if i]
        freq = pd.Series(tous_interets).value_counts()
        top_interets = freq.index.tolist()   # intérêts triés du plus au moins populaire
        populaires = []
        for interet in top_interets:
            if interet in RECOMMANDATIONS_PAR_INTERET:
                populaires.extend(RECOMMANDATIONS_PAR_INTERET[interet][:2])
            if len(populaires) >= 10:
                break
        return populaires[:10]

    def trouver_similaires(self, profil, top_n=5):
        vecteur_cible = profil.vecteur_interets()
        # Si le vecteur cible est nul, impossible de calculer la similarité cosinus
        if not np.any(vecteur_cible):
            return []
        sims = []
        for autre in self.profils:
            if autre.name == profil.name: continue
            v = autre.vecteur_interets()
            if not np.any(v): continue
            sim = 1 - cosine(vecteur_cible, v)
            # Ignorer les nan (division par zéro dans cosine)
            if np.isnan(sim): continue
            sims.append((autre, round(sim, 3)))
        return sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]

    def recommander(self, profil):
        # Cas sans intérêts : on retourne le top 10 des contenus populaires
        if not profil.interests:
            return {
                'suggestions_personnelles':   [],
                'suggestions_collaboratives': [],
                'utilisateurs_similaires':    [],
                'top10_populaires':           self.top10_populaires,
            }

        # filter / map / reduce sur la liste d'intérêts
        interets_valides = list(filter(lambda i: i in RECOMMANDATIONS_PAR_INTERET, profil.interests))
        listes_reco      = list(map(lambda i: RECOMMANDATIONS_PAR_INTERET[i], interets_valides))
        recos_perso      = reduce(lambda a, b: a + b, listes_reco) if listes_reco else []


        # Recommandations collaboratives
        similaires = self.trouver_similaires(profil, top_n=5)
        recos_collab = []
        if similaires:
            # Étape 1 — Score des intérêts via les activités des similaires
            # (capture ce que les similaires font vraiment, pas seulement leurs intérêts déclarés)
            score_interets = {}
            actions_cible  = set(profil.activity_log)
            for autre_profil, sim_score in similaires:
                # Intérêts déclarés des similaires, pondérés par similarité
                for interet in autre_profil.interests:
                    score_interets[interet] = score_interets.get(interet, 0) + sim_score * 2
                # Intérêts inférés depuis les actions des similaires que le profil n'a pas faites
                for action in autre_profil.activity_log:
                    if action in actions_cible:
                        continue
                    for interet, actions in ACTIVITY_MAPPING.items():
                        if action in actions:
                            score_interets[interet] = score_interets.get(interet, 0) + sim_score
                            break

            interets_tries = sorted(score_interets.items(), key=lambda x: x[1], reverse=True)

            # Étape 2 — Pour chaque intérêt scoré : prendre des contenus pas encore suggérés en perso
            deja_recommandes = set(recos_perso)
            for interet, _ in interets_tries:
                if interet in RECOMMANDATIONS_PAR_INTERET:
                    nouveaux = [r for r in RECOMMANDATIONS_PAR_INTERET[interet] if r not in deja_recommandes]
                    recos_collab.extend(nouveaux[:2])
                    deja_recommandes.update(nouveaux[:2])
                if len(recos_collab) >= 6:
                    break

            # Étape 3 — Fallback : si toujours vide, compléter avec le top populaire global
            if not recos_collab:
                recos_collab = [r for r in self.top10_populaires if r not in set(recos_perso)][:6]

        return {
            'suggestions_personnelles':   recos_perso,
            'suggestions_collaboratives': recos_collab,
            'utilisateurs_similaires':    [(p.name, s) for p, s in similaires[:3]],
            'top10_populaires':           [],   # vide si l'utilisateur a des intérêts
        }

# ── Chargement des données ────────────────────────────────────────────────────
@st.cache_data
def charger_donnees():
    def generate_user():
        age      = int(np.random.randint(18, 66))
        nb       = np.random.randint(1, 4)
        # Intérêts et activités stockés directement en chaîne 'elt1,elt2,...'
        interests    = liste_vers_chaine([str(x) for x in np.random.choice(INTERESTS, size=nb, replace=False, p=INTEREST_PROBS)])
        # Les activités sont tirées sur TOUS les intérêts (pas seulement ceux de l'utilisateur)
        # → crée de la diversité dans les vecteurs → similarités variées entre profils
        activity_log = liste_vers_chaine([
            str(np.random.choice(ACTIVITY_MAPPING[str(np.random.choice(INTERESTS))]))
            for _ in range(np.random.randint(3, 7))
        ])
        return {"name": fake.name(), "age": age, "interests": interests, "activity_log": activity_log}

    rows = [generate_user() for _ in range(500)]
    df   = pd.DataFrame(rows)
    df['age'] = df['age'].astype(float)

    # Injection de bruit
    df.loc[np.random.choice(500, 25, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(500, 15, replace=False), 'activity_log'] = None

    df = pd.concat([df, df.sample(n=10, random_state=42)], ignore_index=True)

    # Nettoyage
    df = df.drop_duplicates(subset=['name', 'age', 'interests', 'activity_log'])
    df = df.dropna(subset=['age', 'activity_log']).reset_index(drop=True)

    # ── Séparation en deux datasets ──────────────────────────────────────────
    # Clé de jointure commune
    df['user_id'] = df.index

    # Dataset 1 : profil utilisateur (données d'identité)
    df_utilisateurs = df[['user_id', 'name', 'age']].copy().reset_index(drop=True)

    # Dataset 2 : activités et intérêts (données comportementales)
    df_activites = df[['user_id', 'interests', 'activity_log']].copy().reset_index(drop=True)

    return df_utilisateurs, df_activites

df_utilisateurs, df_activites = charger_donnees()

# ── Jointure pour reconstituer le DataFrame complet utilisé par le reste de l'app ──
df = df_utilisateurs.merge(df_activites, on='user_id').drop(columns=['user_id'])

if 'moteur' not in st.session_state:
    st.session_state.moteur = MoteurRecommandation(df)
if 'df' not in st.session_state:
    st.session_state.df = df
if 'df_utilisateurs' not in st.session_state:
    st.session_state.df_utilisateurs = df_utilisateurs
if 'df_activites' not in st.session_state:
    st.session_state.df_activites = df_activites

moteur = st.session_state.moteur
df     = st.session_state.df

# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🎯 Générateur de contenu personnalisé basé sur l'IA")
st.caption("NumPy · Pandas · SciPy · Matplotlib · Seaborn · POO · Similarité cosinus")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("", [
        "🏠 Recommandations",
        "➕ Ajouter un utilisateur",
        "📊 Analyse statistique",
        "📈 Visualisations",
    ])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — RECOMMANDATIONS
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Recommandations":
    st.subheader("Recommandations personnalisées")

    noms = [p.name for p in moteur.profils]
    nom_selectionne = st.selectbox("Choisir un utilisateur", noms)

    profil = next(p for p in moteur.profils if p.name == nom_selectionne)
    recos  = moteur.recommander(profil)

    # .interests et .activity_log retournent des listes via les propriétés
    st.markdown(f"**Âge :** {profil.age} ans &nbsp;|&nbsp; **Intérêts :** {', '.join(profil.interests) if profil.interests else '_(aucun)_'}")
    st.markdown(f"**Journal d'activité :** {', '.join(profil.activity_log) if profil.activity_log else '_(aucune activité)_'}")
    st.divider()

    # Cas sans intérêts : afficher le top 10 populaire
    if recos['top10_populaires']:
        st.info("ℹ️ Aucun intérêt renseigné — voici les **10 contenus les plus populaires** du moment :")
        for r in recos['top10_populaires']:
            st.markdown(f"- {r}")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🎯 Suggestions personnalisées")
            st.caption("Basées sur vos intérêts et activités")
            for r in recos['suggestions_personnelles']:
                st.markdown(f"- {r}")

        with col2:
            st.markdown("#### 👥 Découvertes collaboratives")
            st.caption("Basées sur les actions des profils similaires")
            if recos['suggestions_collaboratives']:
                for r in recos['suggestions_collaboratives']:
                    st.markdown(f"- {r}")
            else:
                st.info("Aucune découverte collaborative pour le moment.")

        with col3:
            st.markdown("#### 🔗 Profils similaires")
            st.caption("Utilisateurs avec des goûts proches (similarité cosinus)")
            if recos['utilisateurs_similaires']:
                for nom, score in recos['utilisateurs_similaires']:
                    st.markdown(f"- **{nom}** *(score : {score})*")
            else:
                st.info("Aucun profil similaire trouvé.")

    premium = ProfilPremium(profil.name, profil.age, profil.interests, profil.activity_log)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — AJOUTER UN UTILISATEUR
# ════════════════════════════════════════════════════════════════════════════
elif page == "➕ Ajouter un utilisateur":
    st.subheader("Ajouter un nouvel utilisateur")

    col1, col2 = st.columns(2)
    with col1:
        nouveau_nom = st.text_input("Nom", placeholder="ex. Kouassi Akré")
        nouveau_age = st.slider("Âge", 18, 65, 25)
    with col2:
        nouveaux_interets = st.multiselect("Intérêts", options=INTERESTS, default=["fitness"])

    actions_disponibles = []
    for i in nouveaux_interets:
        actions_disponibles.extend(ACTIVITY_MAPPING.get(i, []))

    nouvelles_actions = st.multiselect(
        "Journal d'activité (actions cohérentes avec vos intérêts)",
        options=actions_disponibles
    )

    if st.button("✅ Ajouter et générer les recommandations"):
        if not nouveau_nom:
            st.error("Le nom est obligatoire.")
        else:
            # On passe des listes ; le constructeur les convertit en chaîne
            nouveau_profil = ProfilUtilisateur(nouveau_nom, nouveau_age, nouveaux_interets, nouvelles_actions)
            st.session_state.moteur.profils.append(nouveau_profil)

            # Nouvel identifiant
            nouveau_user_id = st.session_state.df_utilisateurs['user_id'].max() + 1 if len(st.session_state.df_utilisateurs) > 0 else 0

            # Mise à jour du dataset utilisateurs
            nouvelle_ligne_user = pd.DataFrame([{
                'user_id': nouveau_user_id,
                'name':    nouveau_nom,
                'age':     float(nouveau_age),
            }])
            st.session_state.df_utilisateurs = pd.concat([st.session_state.df_utilisateurs, nouvelle_ligne_user], ignore_index=True)

            # Mise à jour du dataset activités
            nouvelle_ligne_activite = pd.DataFrame([{
                'user_id':      nouveau_user_id,
                'interests':    liste_vers_chaine(nouveaux_interets),
                'activity_log': liste_vers_chaine(nouvelles_actions),
            }])
            st.session_state.df_activites = pd.concat([st.session_state.df_activites, nouvelle_ligne_activite], ignore_index=True)

            # Jointure pour maintenir df synchronisé
            nouvelle_ligne = pd.DataFrame([{
                'name':         nouveau_nom,
                'age':          float(nouveau_age),
                'interests':    liste_vers_chaine(nouveaux_interets),
                'activity_log': liste_vers_chaine(nouvelles_actions),
            }])
            st.session_state.df = pd.concat([st.session_state.df, nouvelle_ligne], ignore_index=True)

            recos = st.session_state.moteur.recommander(nouveau_profil)
            st.success(f"✅ **{nouveau_nom}** ajouté avec succès !")
            st.divider()

            # Cas sans intérêts : afficher le top 10 populaire
            if recos['top10_populaires']:
                st.info("ℹ️ Aucun intérêt renseigné — voici les **10 contenus les plus populaires** du moment :")
                for r in recos['top10_populaires']:
                    st.markdown(f"- {r}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("#### 🎯 Suggestions personnalisées")
                    st.caption("Basées sur vos intérêts et activités")
                    for r in recos['suggestions_personnelles']:
                        st.markdown(f"- {r}")
                with col2:
                    st.markdown("#### 👥 Découvertes collaboratives")
                    st.caption("Basées sur les actions des profils similaires")
                    if recos['suggestions_collaboratives']:
                        for r in recos['suggestions_collaboratives']:
                            st.markdown(f"- {r}")
                    else:
                        st.info("Aucune découverte collaborative pour le moment.")
                with col3:
                    st.markdown("#### 🔗 Profils similaires")
                    st.caption("Utilisateurs avec des goûts proches (similarité cosinus)")
                    if recos['utilisateurs_similaires']:
                        for nom, score in recos['utilisateurs_similaires']:
                            st.markdown(f"- **{nom}** *(score : {score})*")
                    else:
                        st.info("Aucun profil similaire trouvé.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYSE STATISTIQUE
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse statistique":
    st.subheader("Analyse statistique")

    df_courant = st.session_state.df.copy()
    # interests et activity_log sont des chaînes 'elt1,elt2,...' → on split pour itérer
    df_courant['interests_list']    = df_courant['interests'].apply(chaine_vers_liste)
    df_courant['activity_log_list'] = df_courant['activity_log'].apply(chaine_vers_liste)
    logs_clean = df_courant.dropna(subset=['interests', 'activity_log']).reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Utilisateurs", len(df_courant))
    col2.metric("Logs nettoyés", len(logs_clean))
    col3.metric("Intérêts disponibles", len(INTERESTS))

    # ── Aperçu des deux datasets séparés ────────────────────────────────────
    st.divider()
    st.markdown("#### 📁 Datasets séparés")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**`df_utilisateurs`** — Profils (identité)")
        st.dataframe(st.session_state.df_utilisateurs.head(10), use_container_width=True)
    with col_b:
        st.markdown("**`df_activites`** — Activités & intérêts")
        st.dataframe(st.session_state.df_activites.head(10), use_container_width=True)
    st.divider()

    # Chi² distribution des intérêts vs Dirichlet
    st.markdown("#### Test χ² — Distribution des intérêts (vs Dirichlet)")
    tous_interets = [i for liste in df_courant['interests_list'] for i in liste]
    freq_interets = pd.Series(tous_interets).value_counts()
    observees     = [freq_interets.get(i, 0) for i in INTERESTS]
    total         = sum(observees)
    attendues     = [total * p for p in INTEREST_PROBS]
    attendues_int = [int(round(x)) for x in attendues]
    attendues_int[-1] += total - sum(attendues_int)

    df_chi = pd.DataFrame({'Intérêt': INTERESTS, 'Observé': observees, 'Attendu': attendues_int})
    st.dataframe(df_chi.set_index('Intérêt'))

    chi2, p_valeur = stats.chisquare(observees, attendues_int)
    st.markdown(f"**χ² = {chi2:.2f}** | **p = {p_valeur:.4f}**")
    if p_valeur < 0.05:
        st.success("→ Distribution NON uniforme : certains intérêts sont significativement plus populaires ✅")
    else:
        st.info("→ Distribution conforme aux probabilités Dirichlet attendues")
    st.divider()

    # Chi² de contingence
    st.markdown("#### Test χ² de contingence — Regarder du contenu IA → Acheter tech")
    df_test = logs_clean.copy()
    df_test['a_vu_ia']       = df_test['activity_log'].apply(lambda s: 1 if "regardé une conférence sur l'IA" in chaine_vers_liste(s) else 0)
    df_test['a_achete_ordi'] = df_test['activity_log'].apply(lambda s: 1 if 'acheté un ordinateur portable' in chaine_vers_liste(s) else 0)
    tableau = pd.crosstab(df_test['a_vu_ia'], df_test['a_achete_ordi'])
    st.dataframe(tableau)
    if tableau.shape == (2, 2):
        chi2_c, p_c, dof, _ = stats.chi2_contingency(tableau)
        st.markdown(f"**χ² = {chi2_c:.2f}** | **p = {p_c:.4f}** | dof = {dof}")
        if p_c < 0.05:
            st.success("→ Relation statistiquement significative ✅")
        else:
            st.warning("→ Aucune relation significative entre ces deux actions")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualisations":
    st.subheader("Visualisations")
    sns.set_theme(style="whitegrid")

    df_courant = st.session_state.df.copy()
    df_courant = df_courant.dropna(subset=['interests', 'activity_log']).reset_index(drop=True)
    # Colonnes listes construites à partir des chaînes pour les visualisations
    df_courant['interests_list']    = df_courant['interests'].apply(chaine_vers_liste)
    df_courant['activity_log_list'] = df_courant['activity_log'].apply(chaine_vers_liste)

    tous_interets = [i for liste in df_courant['interests_list'] for i in liste]
    freq_interets = pd.Series(tous_interets).value_counts()

    tab1, tab2, tab3 = st.tabs([
        "Répartition des intérêts",
        "Heatmap activité × intérêt",
        "Recommandations par segment"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        couleurs = sns.color_palette('Set2', len(freq_interets))
        ax.bar(freq_interets.index, freq_interets.values, color=couleurs, edgecolor='white')
        for i, v in enumerate(freq_interets.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=10)
        ax.set_title("Répartition des centres d'intérêt des utilisateurs", fontsize=13, fontweight='bold')
        ax.set_xlabel("Intérêt")
        ax.set_ylabel("Nombre d'utilisateurs")
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        def detecter_type(action):
            if action.startswith('regardé'):  return 'regardé'
            if action.startswith('acheté'):   return 'acheté'
            if action.startswith('aimé'):     return 'aimé'
            if action.startswith('lu'):       return 'lu'
            if action.startswith('partagé'):  return 'partagé'
            if action.startswith('suivi'):    return 'suivi'
            return 'autre'

        lignes = []
        for _, row in df_courant.iterrows():
            for interet in row['interests_list']:
                for action in row['activity_log_list']:
                    lignes.append({'interet': interet, 'type_action': detecter_type(action)})

        df_hm   = pd.DataFrame(lignes)
        matrice = df_hm.groupby(['type_action', 'interet']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(matrice, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5, ax=ax,
                    cbar_kws={'label': "Nombre d'occurrences"})
        ax.set_title("Intensité d'activité par type d'action et catégorie d'intérêt", fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        df_seg = df_courant.copy()
        df_seg['segment'] = pd.cut(df_seg['age'], bins=[17, 25, 35, 50, 65],
                                    labels=['18-25', '26-35', '36-50', '51-65'])
        comptage = {seg: {i: 0 for i in INTERESTS} for seg in ['18-25', '26-35', '36-50', '51-65']}
        for _, row in df_seg.iterrows():
            seg = str(row['segment'])
            if seg == 'nan': continue
            for interet in row['interests_list']:
                if interet in comptage[seg]:
                    comptage[seg][interet] += 1
        df_plot = pd.DataFrame(comptage).T
        df_plot.index.name = "Segment d'âge"
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot.plot(kind='bar', ax=ax, colormap='tab10', edgecolor='white')
        ax.set_title("Catégories recommandées par segment d'utilisateurs", fontsize=13, fontweight='bold')
        ax.set_xlabel("Segment d'âge")
        ax.set_ylabel("Nombre de recommandations")
        ax.legend(title='Intérêt', bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        st.pyplot(fig)