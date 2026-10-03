from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from core import BatchProcessor, ProcessConfig, FaceDetector

APP_TITLE = '批量二寸证件照生成器 v1.2'


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry('840x760')
        self.minsize(800, 720)
        self.stop_flag = False
        self.worker = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.face_ratio_var = tk.DoubleVar(value=45.0)
        self.center_y_var = tk.DoubleVar(value=42.0)
        self.straighten_var = tk.BooleanVar(value=True)
        self.blur_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=False)
        self.review_var = tk.BooleanVar(value=True)
        self.skip_detection_errors_var = tk.BooleanVar(value=True)

        self._build()
        self._update_detector_label()

    def _build(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill='both', expand=True)

        title = ttk.Label(root, text='批量二寸证件照生成器', font=('Microsoft YaHei UI', 18, 'bold'))
        title.pack(anchor='w')
        ttk.Label(root, text='自动识别人脸 · 自动居中 · 自动轻微纠斜 · 35×49mm · 413×579px · 300DPI').pack(anchor='w', pady=(4, 14))

        paths = ttk.LabelFrame(root, text='1. 文件夹', padding=12)
        paths.pack(fill='x')
        self._path_row(paths, '原始照片：', self.input_var, self.choose_input, 0)
        self._path_row(paths, '输出目录：', self.output_var, self.choose_output, 1)

        opts = ttk.LabelFrame(root, text='2. 构图设置', padding=12)
        opts.pack(fill='x', pady=12)

        ttk.Label(opts, text='输出规格').grid(row=0, column=0, sticky='w')
        ttk.Label(opts, text='二寸 35×49mm / 413×579px / 300DPI').grid(row=0, column=1, sticky='w', padx=(12, 0))

        ttk.Label(opts, text='人脸高度占成片').grid(row=1, column=0, sticky='w', pady=(12, 0))
        s1 = ttk.Scale(opts, from_=38, to=55, variable=self.face_ratio_var, orient='horizontal', length=260, command=lambda _v: self._refresh_values())
        s1.grid(row=1, column=1, sticky='w', padx=(12, 0), pady=(12, 0))
        self.face_ratio_label = ttk.Label(opts, width=6)
        self.face_ratio_label.grid(row=1, column=2, pady=(12, 0))

        ttk.Label(opts, text='人脸中心垂直位置').grid(row=2, column=0, sticky='w', pady=(10, 0))
        s2 = ttk.Scale(opts, from_=38, to=48, variable=self.center_y_var, orient='horizontal', length=260, command=lambda _v: self._refresh_values())
        s2.grid(row=2, column=1, sticky='w', padx=(12, 0), pady=(10, 0))
        self.center_y_label = ttk.Label(opts, width=6)
        self.center_y_label.grid(row=2, column=2, pady=(10, 0))
        self._refresh_values()

        checks = ttk.Frame(opts)
        checks.grid(row=3, column=0, columnspan=3, sticky='w', pady=(12, 0))
        ttk.Checkbutton(checks, text='根据双眼自动轻微纠斜', variable=self.straighten_var).pack(side='left')
        ttk.Checkbutton(checks, text='标记低清晰度照片', variable=self.blur_var).pack(side='left', padx=14)
        ttk.Checkbutton(checks, text='包含子文件夹', variable=self.recursive_var).pack(side='left')

        review_options = ttk.Frame(opts)
        review_options.grid(row=4, column=0, columnspan=3, sticky='w', pady=(8, 0))
        ttk.Checkbutton(review_options, text='开启人工复核', variable=self.review_var).pack(side='left')
        ttk.Label(review_options, text='关闭时不复制复核照片，异常原因仍记录在报告中。').pack(side='left', padx=12)

        anomaly_options = ttk.Frame(opts)
        anomaly_options.grid(row=5, column=0, columnspan=3, sticky='w', pady=(8, 0))
        ttk.Checkbutton(anomaly_options, text='跳过识别异常照片', variable=self.skip_detection_errors_var).pack(side='left')
        ttk.Label(anomaly_options, text='取消勾选：按原图居中裁剪输出二寸照片；损坏图片仍跳过。', wraplength=470).pack(side='left', padx=12)

        detector_box = ttk.LabelFrame(root, text='3. 人脸检测', padding=12)
        detector_box.pack(fill='x')
        self.detector_label = ttk.Label(detector_box, text='')
        self.detector_label.pack(anchor='w')
        ttk.Label(
            detector_box,
            text='提示：models 文件夹中存在 YuNet 模型时优先使用；缺失时会自动使用 OpenCV Haar 备用检测器。',
            wraplength=690,
        ).pack(anchor='w', pady=(4, 0))

        run = ttk.LabelFrame(root, text='4. 批量处理', padding=12)
        run.pack(fill='both', expand=True, pady=(12, 0))
        btns = ttk.Frame(run)
        btns.pack(fill='x')
        self.start_btn = ttk.Button(btns, text='开始批量处理', command=self.start)
        self.start_btn.pack(side='left')
        self.stop_btn = ttk.Button(btns, text='停止', command=self.stop, state='disabled')
        self.stop_btn.pack(side='left', padx=8)
        ttk.Button(btns, text='打开输出目录', command=self.open_output).pack(side='left')

        self.progress = ttk.Progressbar(run, mode='determinate')
        self.progress.pack(fill='x', pady=(12, 6))
        self.status_label = ttk.Label(run, text='等待开始')
        self.status_label.pack(anchor='w')

        self.log = tk.Text(run, height=10, wrap='word', font=('Consolas', 9))
        self.log.pack(fill='both', expand=True, pady=(8, 0))
        self.log.insert('end', '成品保留原始子目录结构；开启人工复核后才复制复核照片。详情见处理报告.csv\n')
        self.log.configure(state='disabled')

    def _path_row(self, parent, label, var, command, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky='ew', padx=8, pady=4)
        ttk.Button(parent, text='选择', command=command).grid(row=row, column=2, pady=4)
        parent.columnconfigure(1, weight=1)

    def _refresh_values(self):
        if hasattr(self, 'face_ratio_label'):
            self.face_ratio_label.configure(text=f'{self.face_ratio_var.get():.0f}%')
        if hasattr(self, 'center_y_label'):
            self.center_y_label.configure(text=f'{self.center_y_var.get():.0f}%')

    def _update_detector_label(self):
        try:
            detector = FaceDetector()
            if detector.kind == 'YuNet':
                text = f'当前：YuNet（推荐，模型已验证） | {detector.model_path.name}'
            elif detector.load_warning:
                text = '当前：OpenCV Haar（备用） | YuNet 加载失败，详情见日志。'
                self.append_log(detector.load_warning)
            else:
                text = '当前：OpenCV Haar（备用） | 可运行 download_model.py 安装 YuNet。'
            self.detector_label.configure(text=text, wraplength=690)
        except Exception as exc:
            self.detector_label.configure(text='人脸检测器初始化失败，详情见日志。')
            self.append_log(str(exc))

    def choose_input(self):
        p = filedialog.askdirectory(title='选择原始照片文件夹')
        if p:
            self.input_var.set(p)
            if not self.output_var.get():
                self.output_var.set(str(Path(p).parent / f'{Path(p).name}_证件照输出'))

    def choose_output(self):
        p = filedialog.askdirectory(title='选择输出目录')
        if p:
            self.output_var.set(p)

    def open_output(self):
        p = Path(self.output_var.get().strip()) if self.output_var.get().strip() else None
        if not p or not p.exists():
            messagebox.showinfo('提示', '输出目录还不存在。')
            return
        try:
            os.startfile(p)  # Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(['xdg-open', str(p)])

    def append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def start(self):
        in_p = Path(self.input_var.get().strip())
        out_p = Path(self.output_var.get().strip())
        if not self.input_var.get().strip() or not in_p.is_dir():
            messagebox.showerror('错误', '请选择有效的原始照片文件夹。')
            return
        if not self.output_var.get().strip():
            messagebox.showerror('错误', '请选择输出目录。')
            return
        if in_p.resolve() == out_p.resolve():
            messagebox.showerror('错误', '输出目录不能与原始照片目录相同。')
            return
        try:
            out_p.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror('错误', f'无法创建输出目录：{exc}')
            return

        self.stop_flag = False
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.progress['value'] = 0
        self.status_label.configure(text='正在准备……')

        cfg = ProcessConfig(
            face_height_ratio=self.face_ratio_var.get() / 100.0,
            face_center_y_ratio=self.center_y_var.get() / 100.0,
            auto_straighten=self.straighten_var.get(),
            flag_blur=self.blur_var.get(),
            recursive=self.recursive_var.get(),
            enable_review=self.review_var.get(),
            skip_detection_errors=self.skip_detection_errors_var.get(),
        )
        self.worker = threading.Thread(target=self._run_worker, args=(in_p, out_p, cfg), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag = True
        self.status_label.configure(text='正在停止……')

    def _run_worker(self, in_p: Path, out_p: Path, cfg: ProcessConfig):
        try:
            processor = BatchProcessor(cfg)
            self.after(0, lambda: self.append_log(f'检测器：{processor.detector.kind}'))

            def progress(i, total, result):
                def ui():
                    pct = 0 if total == 0 else i / total * 100
                    self.progress['value'] = pct
                    self.status_label.configure(text=f'{i}/{total}  {result.filename}  →  {result.status}')
                    self.append_log(f'[{i}/{total}] {result.filename}: {result.status} - {result.reason}')
                self.after(0, ui)

            results, report = processor.process_folder(
                in_p, out_p,
                progress_cb=progress,
                stop_cb=lambda: self.stop_flag,
            )
            done = sum(1 for r in results if r.status.startswith('完成'))
            review = sum(1 for r in results if r.status == '复核')
            warn = sum(1 for r in results if r.status == '完成/警告')
            skipped = sum(1 for r in results if r.status in ('跳过', '复核'))
            review_summary = f'待复核 {review + warn}' if cfg.enable_review else '人工复核关闭'

            def finish():
                self.start_btn.configure(state='normal')
                self.stop_btn.configure(state='disabled')
                if self.stop_flag:
                    self.status_label.configure(text=f'已停止。已处理 {len(results)} 张。')
                else:
                    self.progress['value'] = 100
                    self.status_label.configure(text=f'完成：输出 {done}，跳过 {skipped}，{review_summary}，输出警告 {warn}')
                    messagebox.showinfo('处理完成', f'已处理 {len(results)} 张\n成功输出：{done}\n跳过输出：{skipped}\n{review_summary}\n输出警告：{warn}\n报告：{report}')
            self.after(0, finish)
        except Exception as exc:
            def fail(exc=exc):
                self.start_btn.configure(state='normal')
                self.stop_btn.configure(state='disabled')
                self.status_label.configure(text='处理失败')
                self.append_log(f'错误：{exc!r}')
                messagebox.showerror('处理失败', str(exc))
            self.after(0, fail)


if __name__ == '__main__':
    App().mainloop()
