-- FragPredict: 'predictions' tablosu için en az yetki.
-- API publishable anahtarla (sb_publishable_ -> 'anon' rolü) bağlanır ve SADECE satır
-- ekleyebilir: okuma, güncelleme, silme yok. Anahtar sızsa bile yapılabilecek tek şey,
-- herkese açık API'nin zaten yaptığı şey olur: satır eklemek.
-- Supabase Dashboard -> SQL Editor'de bir kez çalıştırılır; tekrar çalıştırmak güvenli.

alter table public.predictions enable row level security;

-- Tablo yetkileri: anon sadece INSERT yapabilsin (Supabase varsayılanı ALL veriyor)
revoke all on table public.predictions from anon, authenticated;
grant insert on table public.predictions to anon;

-- RLS kuralı: anon satır ekleyebilir, olasılıklar 0-1 aralığında olmak şartıyla
drop policy if exists "anon_insert_only" on public.predictions;
create policy "anon_insert_only" on public.predictions
    for insert to anon
    with check (
        team1_win_probability between 0 and 1
        and team2_win_probability between 0 and 1
    );

-- Kontrol: tablodaki tüm kurallar. "anon_insert_only" dışında anon/public rolüne
-- SELECT/UPDATE/DELETE veren bir kural görünüyorsa kaldırılmalı.
select policyname, cmd, roles
from pg_policies
where schemaname = 'public' and tablename = 'predictions';
