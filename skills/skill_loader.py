import os

def load_skill(skill_path: str) -> str:
    """Charge un fichier .md de skill et retourne son contenu."""
    if not os.path.exists(skill_path):
        print(f"⚠️  Skill non trouvé : {skill_path}")
        return ""
    
    with open(skill_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    print(f"✅ Skill chargé : {skill_path}")
    return content