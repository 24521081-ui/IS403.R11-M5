"""Finish only after validation: select, conditional test, verify, groups, figures, handoff."""
from pathlib import Path
import json
import subprocess
import sys
import os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/analysis/c7_events_20261126'
os.environ.setdefault('PYTHONUTF8','1')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')

def execute(script,*args):
    command=[sys.executable,str(ROOT/'scripts'/script),*args]
    with (OUT/'audit/commands.txt').open('a',encoding='utf-8') as stream:
        stream.write(subprocess.list2cmdline(command)+'\n')
    subprocess.run(command,check=True,cwd=ROOT)

def main():
    assert len(list((OUT/'validation').glob('*/completed.json')))==10, 'Validation incomplete; group scoring forbidden.'
    (OUT/'audit').mkdir(parents=True, exist_ok=True)
    execute('run_c7_events.py','--stage','select')
    selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']
    if selected=='C7_EVENTS':execute('run_c7_events.py','--stage','test')
    execute('run_c7_events.py','--stage','verify-test')
    execute('run_c7_events.py','--stage','groups')
    execute('plot_c7_events.py')
    import pandas as pd
    comparison=pd.read_csv(OUT/'validation_comparison.csv')
    scores=pd.read_csv(OUT/'test_group_scores.csv')
    provenance_path=OUT/'audit/provenance.json'
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    else:
        provenance={key:subprocess.check_output(command,cwd=ROOT,text=True).strip()
                    for key,command in [('branch',['git','branch','--show-current']),
                                        ('head',['git','rev-parse','HEAD'])]}
        provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    verified=json.loads((OUT/'test_verified.json').read_text(encoding='utf-8'))
    labels=json.loads((OUT/'labels_audit.json').read_text(encoding='utf-8'))
    cfg=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    inventory=[]
    for path in sorted((OUT/'models').glob('*/d_*/training.json')):
        info=json.loads(path.read_text(encoding='utf-8'))
        assert len(info['features'])==48
        assert info['actual_trees']=={'classifier':300,'regressor':200}
        assert info['parameters']==cfg['parameters']
        inventory.append(dict(store=path.parent.parent.name,origin=info['fit_end_day'],
            rows=info['n_train_rows'],positive_rows=info['n_positive_rows'],features=len(info['features']),
            classifier_trees=info['actual_trees']['classifier'],regressor_trees=info['actual_trees']['regressor']))
    pd.DataFrame(inventory).to_csv(OUT/'audit/training_inventory.csv',index=False)
    lines=['BÀN GIAO THÍ NGHIỆM C7 THÊM SỰ KIỆN — 26/11/2026',
        f"Nhánh làm việc: {provenance['branch']} tại {provenance['head']}.",
        'Nguồn mã và hướng dẫn tái lập: docs/C7_EVENTS.md; script bàn giao không fetch hoặc push.',
        'Không có main.py trên origin/main. run.py không gọi script nhãn. Script nhãn có main(argv) riêng.',
        'Lệnh mặc định run_all_experiments.py chứa đường dẫn cố định và chạy cấu hình khác; không dùng.',
        '', 'ĐẶC TRƯNG VÀ QUY TRÌNH',
        'Cả Classifier và Regressor C7 gốc dùng: '+', '.join(cfg['original_features'])+'.',
        'Đã có day_of_week, month, event_flag và SNAP theo bang; event_flag không phải đặc trưng mới.',
        'Thêm 30 chỉ báo tên sự kiện và 4 chỉ báo loại, gộp hai slot calendar; không thay bộ giá hoặc siêu tham số.',
        'Tên biến và ánh xạ đầy đủ: config.json. Từ điển cố định từ lịch đến d_1857.',
        'Lịch/sự kiện tương lai theo giả định lịch ngoại sinh được cung cấp sẵn của M5; không dùng doanh số hoặc giá holdout.',
        'Classifier: 300 cây, 48 lá; Regressor: 200 cây, 31 lá; learning_rate=0.05; seed=42; 6 luồng CPU.',
        'Cả hai thành phần được train lại. Giá tuần hoàn tất trước đó; giữ giá tại origin cho 28 ngày dự báo.',
        'audit/training_inventory.csv ghi số dòng, số biến và số cây thực tế của từng mô hình mới.',
        'Validation: auxiliary ≤1857 → OOF 1858–1885 → fit Platt(logit(p_raw)), C=1, lbfgs, max_iter=1000.',
        'Main ≤1885 → dự báo đệ quy 1886–1913, p_blend=0.25*p_raw+0.75*p_cal; yhat=p_blend*mu.',
        'Test: auxiliary ≤1885 → OOF 1886–1913 → fit Platt; main ≤1913 → dự báo 1914–1941.',
        'Lưu ý: calendar.csv không có ngày sự kiện trong validation 1886–1913. Chênh lệch phản ánh tác động fit và calibration, chưa đánh giá trực tiếp ngày lễ.',
        'OOF validation có StPatricksDay, Purim End, Easter. Test có Pesach End, OrthodoxEaster, Cinco De Mayo, Mother\'s day.',
        'Không có NBAFinals trong cả ba cửa sổ dự báo; các chỉ báo đó chỉ xuất hiện trong lịch sử train. Xem audit/events_in_forecast_windows.json.',
        '', 'ĐỐI CHỨNG VALIDATION (cùng score_c7_wrmsse.score, WRMSSE 12 cấp)',comparison.to_string(index=False),
        f'Cấu hình chọn: {selected}. Chọn thêm sự kiện chỉ khi WRMSSE validation thấp hơn; bằng hoặc cao hơn giữ C7.',
        'Điểm cấp 12 là RMSSE trọng số doanh thu của cấp đó, chưa chia 12. Không dùng test để chọn cấu hình.',
        '', 'TEST ĐÃ XÁC MINH',json.dumps(verified,ensure_ascii=False,indent=2),
        '', 'NHÓM CỐ ĐỊNH TỪ LỊCH SỬ ĐẾN d_1885',json.dumps(labels,ensure_ascii=False,indent=2),
        'ADI = số ngày từ lần bán đầu đến origin / số ngày có bán. ADI ≤1.32: frequent; còn lại intermittent.',
        'Chưa từng bán hoặc dưới 2 ngày có bán: intermittent theo script nhóm. Scale RMSSE theo scorer M5 hiện tại.',
        'Trọng số test = doanh thu thực tế 1886–1913; chuẩn hóa trong từng nhóm. RMSSE tại item × store (cấp 12).',
        'Sai lệch lượng = 100*(tổng dự báo/tổng thực tế-1); dương dư, âm thiếu.',scores.to_string(index=False),
        'Chỉ phân tích cấu hình C7 được chọn; chưa có đủ sáu bộ dự báo test hợp lệ để vẽ so sánh sáu mô hình.',
        'Điểm xếp hạng chung vẫn là WRMSSE 12 cấp; test_levels.csv lưu toàn bộ 12 cấp.',
        '', 'HÌNH SVG/PNG',*[str(p.relative_to(ROOT)) for p in sorted((OUT/'figures').iterdir())],
        '', 'LỆNH ĐÃ CHẠY',(OUT/'audit/commands.txt').read_text(encoding='utf-8'),
        'Kiểm thử mã nguồn: chạy pytest theo docs/C7_EVENTS.md; không suy ra kết quả test từ sự tồn tại của artifact.',
        'Đầu ra cũ giữ nguyên; các thay đổi cục bộ ban đầu có bản diff trong audit. Không commit, không push.']
    (OUT/'BAN_GIAO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(comparison.to_string(index=False))
    print(scores.to_string(index=False))
    print('Handoff:',OUT/'BAN_GIAO.txt')

if __name__=='__main__':main()
