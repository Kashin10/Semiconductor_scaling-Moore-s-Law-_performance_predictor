import pandas as pd

df = pd.read_json("gpus.json")

rows = []

for _, row in df.iterrows():
    gpu = row.get("Name")

    # 🔥 KEY FIX: Settings is wrapped
    settings_obj = row.get("Settings", {})
    
    # sometimes wrapped like {"Value": {...}}
    if isinstance(settings_obj, dict) and "Value" in settings_obj:
        settings = settings_obj["Value"]
    else:
        settings = settings_obj

    if not isinstance(settings, dict):
        continue

    ultra = settings.get("ultra", {})
    resolution_block = ultra.get("Resolution", {})

    for res, res_data in resolution_block.items():

        games_list = res_data.get("Games", [])

        for g in games_list:
            game_name = g.get("Game_Name")
            fps_val = g.get("Avg_FPS")

            if game_name and fps_val:
                fps_clean = str(fps_val).replace(",","")
                rows.append({
                    "gpu_name": gpu,
                    "resolution": res,
                    "game": game_name,
                    "fps": float(fps_clean)
                })

fps_df = pd.DataFrame(rows)

print("Shape:", fps_df.shape)
print(fps_df.head())

fps_df.to_csv("fps_flat.csv", index=False)

print("✅ FPS extraction successful")