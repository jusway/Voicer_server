// Placeholder JS
console.log('Voicer static loaded');



// Upload progress handling
(function(){
  document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form[action="/api/audio/upload"]') || document.getElementById('upload-form');
    if (!form) return;
    const prog = document.getElementById('prog');
    const pct  = document.getElementById('pct');
    const spd  = document.getElementById('spd');
    const res  = document.getElementById('res');
    // Helpers for human readable sizes/speeds
    function humanBytes(b){
      const u=['B','KB','MB','GB','TB']; let i=0; let v=Number(b)||0;
      while(v>=1024 && i<u.length-1){ v/=1024; i++; }
      return (i? v.toFixed(1): v.toFixed(0)) + ' ' + u[i];
    }
    function humanRate(bps){ return humanBytes(bps) + '/s'; }


    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const fd = new FormData(form);
      const xhr = new XMLHttpRequest();

      if (prog) prog.value = 0;
      if (pct)  pct.textContent = '0%';
      if (spd)  spd.textContent = '0 KB/s';
      if (res)  res.textContent = '开始上传...';

      // Sliding window for smoothing speed; throttle UI to 250ms
      const samples = [];
      let lastUi = 0; // seconds

      xhr.upload.onprogress = (ev) => {
        const now = performance.now() / 1000;
        samples.push([now, ev.loaded]);
        while (samples.length && (now - samples[0][0] > 2)) samples.shift();

        if (ev.lengthComputable) {
          if (prog) {
            if (!prog.max || prog.max === 100) prog.max = ev.total; // switch to byte-based progress
            prog.value = ev.loaded;
          }
          if (pct) pct.textContent = (ev.loaded / ev.total * 100).toFixed(1) + '%';
        } else if (pct) {
          pct.textContent = humanBytes(ev.loaded);
        }

        if ((now - lastUi) >= 0.25) {
          lastUi = now;
          if (samples.length > 1) {
            const [t0, b0] = samples[0];
            const [t1, b1] = samples[samples.length - 1];
            const dt = t1 - t0;
            if (dt > 0 && spd) {
              const bps = (b1 - b0) / dt;
              spd.textContent = humanRate(bps);
              if (res && ev.lengthComputable) {
                const remain = ev.total - ev.loaded;
                const eta = bps > 0 ? (remain / bps).toFixed(1) + 's' : '--';
                res.textContent = `${humanBytes(ev.loaded)} / ${humanBytes(ev.total)} · 约剩 ${eta}`;
              }
            }
          }
        }
      };

      xhr.onload = () => {
        if (res) {
          try {
            const json = JSON.parse(xhr.responseText || '{}');
            res.textContent = '上传完成: ' + (json.saved_path || xhr.responseText || '');
          } catch (e) {
            res.textContent = xhr.responseText || '上传完成';
          }
        }
      };

      xhr.onerror = () => { if (res) res.textContent = '上传失败'; };

      xhr.open('POST', '/api/audio/upload');
      xhr.send(fd);
    });
  });
})();
