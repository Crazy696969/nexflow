import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from sklearn.linear_model import LinearRegression
import io
import os
from pathlib import Path

app = FastAPI(title="NexFlow API", version="3.0")

# CORS ayarları - FRONTEND İÇİN DÜZELTİLDİ
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Herkese açık (geliştirme için)
        "https://nexflow-ui.onrender.com",
        "https://nexflow-lxmq.onrender.com",
        "http://localhost:3000",
        "http://localhost:8081"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Ana sayfa - index.html'yi döndür"""
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return {"name": "NexFlow API", "version": "3.0", "status": "active"}

@app.get("/health")
def health():
    return {"status": "healthy", "message": "Motor çalışıyor"}

def safe_float(value):
    """NaN/Inf değerleri None'a çevir"""
    if pd.isna(value) or np.isinf(value):
        return None
    return float(value)

@app.post("/upload-csv")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"Dosya alındı: {file.filename}")
        
        # 1. Dosya boyutu kontrolü (10 MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(413, "Dosya 10 MB'dan büyük olamaz")
        
        # 2. Dosya tipi kontrolü (.csv veya .xlsx)
        filename = file.filename.lower()
        is_excel = filename.endswith('.xlsx') or filename.endswith('.xls')
        is_csv = filename.endswith('.csv')
        
        if not (is_csv or is_excel):
            raise HTTPException(415, "Sadece CSV veya Excel (.xlsx, .xls) dosyaları kabul edilir")
        
        # 3. Dosyayı oku
        try:
            if is_excel:
                df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
            else:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
                    except UnicodeDecodeError:
                        df = pd.read_csv(io.BytesIO(contents), encoding='cp1254')
        except Exception as e:
            raise HTTPException(400, f"Dosya okunamadı: {str(e)}")
        
        print(f"Kolonlar: {list(df.columns)}")
        print(f"Satır sayısı: {len(df)}")
        
        # 4. Zorunlu kolon kontrolü
        if 'satis_adedi' not in df.columns:
            raise HTTPException(422, "CSV'de 'satis_adedi' kolonu bulunamadı")
        
        # 5. satis_adedi'ni sayısal yap, boş/geçersizleri temizle
        df['satis_adedi'] = pd.to_numeric(df['satis_adedi'], errors='coerce')
        df = df.dropna(subset=['satis_adedi'])
        
        # 6. Negatif değerleri filtrele
        negatif_count = (df['satis_adedi'] < 0).sum()
        if negatif_count > 0:
            df = df[df['satis_adedi'] >= 0]
        
        # 7. En az 2 satır kontrolü
        if len(df) < 2:
            raise HTTPException(422, "En az 2 geçerli satır verisi gerekli (tahmin için)")
        
        # 8. Temel istatistikler
        toplam_satis = int(df['satis_adedi'].sum())
        ortalama_satis = safe_float(df['satis_adedi'].mean())
        en_yuksek_satis = int(df['satis_adedi'].max())
        en_dusuk_satis = int(df['satis_adedi'].min())
        
        # 9. Satış serisi (tarih varsa kullan, yoksa index)
        if 'tarih' in df.columns:
            df['tarih'] = pd.to_datetime(df['tarih'], errors='coerce')
            tarih_serisi = df.dropna(subset=['tarih'])['tarih'].dt.strftime('%Y-%m-%d').tolist()
            satis_serisi = df.dropna(subset=['tarih'])['satis_adedi'].tolist()
            if not satis_serisi:
                tarih_serisi = [str(i+1) for i in range(len(df))]
                satis_serisi = df['satis_adedi'].tolist()
        else:
            tarih_serisi = [str(i+1) for i in range(len(df))]
            satis_serisi = df['satis_adedi'].tolist()
        
        # 10. Tahmin (LinearRegression)
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['satis_adedi'].values
        
        if len(X) >= 2 and len(np.unique(y)) > 1:
            model = LinearRegression()
            model.fit(X, y)
            future_X = np.arange(len(df), len(df) + 7).reshape(-1, 1)
            tahmin = model.predict(future_X)
            gelecek_7_gun_tahmini = [float(max(0, round(x))) for x in tahmin]
            trend = "yukselis" if model.coef_[0] > 0 else "dusus"
        else:
            gelecek_7_gun_tahmini = [int(y[-1])] * 7 if len(y) > 0 else [0] * 7
            trend = "bilinmiyor"
        
        # 11. Yanıt yapısı
        response = {
            "toplam_kayit": int(len(df)),
            "sutunlar": list(df.columns),
            "toplam_satis": toplam_satis,
            "ortalama_satis": ortalama_satis,
            "en_yuksek_satis": en_yuksek_satis,
            "en_dusuk_satis": en_dusuk_satis,
            "satis_serisi": {
                "labels": tarih_serisi,
                "values": satis_serisi
            },
            "gelecek_7_gun_tahmini": gelecek_7_gun_tahmini,
            "trend": trend
        }
        
        # 12. Gelir bilgisi (varsa)
        if 'gelir' in df.columns:
            df['gelir'] = pd.to_numeric(df['gelir'], errors='coerce')
            toplam_gelir = safe_float(df['gelir'].sum())
            ortalama_gelir = safe_float(df['gelir'].mean())
            gelir_serisi = df.dropna(subset=['gelir'])['gelir'].tolist()
            
            response.update({
                "toplam_gelir": toplam_gelir,
                "ortalama_gelir": ortalama_gelir,
                "gelir_serisi": {
                    "labels": tarih_serisi[:len(gelir_serisi)],
                    "values": gelir_serisi
                }
            })
            
            # Gelir tahmini
            if len(df) >= 2:
                y_gelir = df['gelir'].dropna().values
                if len(y_gelir) >= 2:
                    X_gelir = np.arange(len(y_gelir)).reshape(-1, 1)
                    model_gelir = LinearRegression()
                    model_gelir.fit(X_gelir, y_gelir)
                    future_X_gelir = np.arange(len(y_gelir), len(y_gelir) + 7).reshape(-1, 1)
                    gelir_tahmin = model_gelir.predict(future_X_gelir)
                    response["gelecek_7_gun_gelir_tahmini"] = [float(max(0, round(x))) for x in gelir_tahmin]
        
        # 13. Ürün bilgisi (varsa)
        if 'urun' in df.columns:
            urun_satis = df.groupby('urun')['satis_adedi'].sum().sort_values(ascending=False)
            en_cok_satan_urun = str(urun_satis.index[0]) if len(urun_satis) > 0 else None
            response.update({
                "en_cok_satan_urun": en_cok_satan_urun,
                "urun_dagilimi": {
                    "labels": [str(u) for u in urun_satis.index],
                    "values": [int(v) for v in urun_satis.values]
                }
            })
        
        print("Analiz başarıyla tamamlandı!")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Hata: {str(e)}")
        raise HTTPException(500, f"Sunucu hatası: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("[NexFlow v3.0] Motor başlatılıyor...")
    uvicorn.run(app, host="0.0.0.0", port=8081)