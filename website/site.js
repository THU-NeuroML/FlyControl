'use strict';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const videoBase = 'assets/videos/';
const posterBase = 'assets/images/';
const videos = [...document.querySelectorAll('video')];
const manuallyPaused = new WeakSet();
const visibleVideos = new Set();

function loadVideo(video) {
  if (!video.dataset.src) return;
  video.src = video.dataset.src;
  delete video.dataset.src;
  video.load();
}
function playVideo(video) {
  if (document.hidden || reducedMotion.matches || manuallyPaused.has(video)) return;
  loadVideo(video);
  const attempt = video.play();
  if (attempt) attempt.catch(() => {});
}
function setVideo(video, stem, label) {
  video.pause();
  video.poster = `${posterBase}${stem}.webp`;
  video.src = `${videoBase}${stem}.mp4`;
  delete video.dataset.src;
  video.setAttribute('aria-label', label);
  video.load();
  if (visibleVideos.has(video)) playVideo(video);
}

const videoObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const video = entry.target;
    if (entry.isIntersecting) {
      visibleVideos.add(video);
      loadVideo(video);
      playVideo(video);
    } else {
      visibleVideos.delete(video);
      video.pause();
    }
  }
}, { threshold: 0.25 });
videos.forEach(video => {
  videoObserver.observe(video);
  video.addEventListener('play', () => { manuallyPaused.delete(video); });
  // Native controls are left intact. Explicit pointer/key interaction on a
  // playing video is remembered across viewport changes when it pauses.
  let userInteraction = false;
  video.addEventListener('pointerdown', () => { userInteraction = true; setTimeout(() => { userInteraction = false; }, 800); });
  video.addEventListener('keydown', () => { userInteraction = true; setTimeout(() => { userInteraction = false; }, 800); });
  video.addEventListener('pause', () => { if (userInteraction) manuallyPaused.add(video); });
});
document.addEventListener('visibilitychange', () => {
  videos.forEach(video => { if (document.hidden) video.pause(); else if (visibleVideos.has(video)) playVideo(video); });
});
reducedMotion.addEventListener('change', () => {
  if (reducedMotion.matches) videos.forEach(video => video.pause());
});

const heroVideo = document.querySelector('#hero-video');
const heroButton = document.querySelector('#hero-play');
heroButton.addEventListener('click', () => {
  loadVideo(heroVideo);
  if (heroVideo.paused) { manuallyPaused.delete(heroVideo); heroVideo.play().catch(() => {}); }
  else { manuallyPaused.add(heroVideo); heroVideo.pause(); }
});
function updateHeroButton() {
  heroButton.textContent = heroVideo.paused ? 'Play ▶' : 'Pause Ⅱ';
  heroButton.setAttribute('aria-label', `${heroVideo.paused ? 'Play' : 'Pause'} hero video`);
}
heroVideo.addEventListener('play', updateHeroButton);
heroVideo.addEventListener('pause', updateHeroButton);

function selectTab(button) {
  const group = button.closest('[role="tablist"]');
  group.querySelectorAll('[role="tab"]').forEach(tab => {
    tab.setAttribute('aria-selected', String(tab === button));
    tab.tabIndex = tab === button ? 0 : -1;
  });
  document.getElementById(button.getAttribute('aria-controls')).setAttribute('aria-labelledby', button.id);
}
document.querySelectorAll('[role="tablist"]').forEach(group => {
  group.addEventListener('keydown', event => {
    const tabs = [...group.querySelectorAll('[role="tab"]')];
    let index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    if (event.key === 'ArrowRight') index = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') index = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') index = 0;
    else if (event.key === 'End') index = tabs.length - 1;
    else return;
    event.preventDefault();
    tabs[index].focus();
    tabs[index].click();
  });
});

const flyTasks = {
  terrain: { index: '01', title: 'Crossing higher obstacles', copy: 'The connectome controller maintains higher all-feet crossing success as block height increases. Both controllers reliably traverse flat ground.', stat: '0.20–0.525', metric: 'mm block heights evaluated', sample: '20 paired training seeds · 64 shared test phases per height', label: 'Block height', figure: '2', conditions: [['fly-terrain','0.525 mm'],['fly_obstacle_0p45','0.45 mm'],['fly_obstacle_0p3','0.30 mm'],['fly_obstacle_0p2','0.20 mm']] },
  turning: { index: '02', title: 'Smoother changes of direction', copy: 'The connectome controller reduces command fluctuations across turns. Terminal heading accuracy varies with turn direction.', stat: '44.6%', metric: 'less total command variation', sample: '24 paired training seeds · six signed target headings', label: 'Target heading', figure: '3', conditions: [['fly-turning','+60°'],['fly_turn_left_30','+30°'],['fly_turn_left_90','+90°'],['fly_turn_right_30','−30°'],['fly_turn_right_60','−60°'],['fly_turn_right_90','−90°']] },
  speed: { index: '03', title: 'Following a changing pace', copy: 'The connectome controller tracks both ramp and sinusoidal commands more accurately, with lower ramp-rate and amplitude-ratio errors.', stat: '8–16', metric: 'mm/s operating range', sample: '20 paired training seeds · ramp and sinusoidal commands', label: 'Speed command', figure: '4', conditions: [['fly-speed','Ramp'],['fly_speed_sine','Sinusoid']] }
};
let flyTask = 'terrain';
const flyCondition = document.querySelector('#fly-condition');
const flyChoices = document.querySelector('#fly-choices');
function renderFlyChoices() {
  const task = flyTasks[flyTask];
  const useSelect = flyTask === 'turning';
  document.querySelector('#fly-select-control').hidden = !useSelect;
  document.querySelector('#fly-choice-control').hidden = useSelect;
  if (useSelect) return;
  document.querySelector('#fly-choice-label').textContent = flyTask === 'terrain' ? 'Block height (mm)' : task.label;
  const choices = flyTask === 'terrain' ? [...task.conditions].reverse() : task.conditions;
  flyChoices.replaceChildren(...choices.map(([value, title]) => {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'fly-choice';
    input.value = value;
    input.checked = value === flyCondition.value;
    const text = document.createElement('span');
    text.textContent = title.replace(' mm', '');
    label.append(input, text);
    return label;
  }));
}
function options(select, values, preferred) {
  select.replaceChildren(...values.map(([value, label]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = label; return option;
  }));
  if (values.some(([value]) => value === preferred)) select.value = preferred;
}
document.querySelectorAll('[data-task]').forEach(button => button.addEventListener('click', () => {
  flyTask = button.dataset.task;
  selectTab(button);
  const task = flyTasks[flyTask];
  options(flyCondition, task.conditions);
  renderFlyChoices();
  document.querySelector('#fly-index').textContent = `${task.index} / 03`;
  document.querySelector('#fly-result-title').textContent = task.title;
  document.querySelector('#fly-result-copy').textContent = task.copy;
  document.querySelector('#fly-stat').textContent = task.stat;
  document.querySelector('#fly-stat-label').textContent = task.metric;
  document.querySelector('#fly-sample').textContent = task.sample;
  document.querySelector('#fly-select-label').textContent = task.label;
  const figureLink = document.querySelector('#fly-figure-link');
  figureLink.dataset.figure = task.figure;
  figureLink.textContent = `View ${flyTask} results ↗`;
  document.querySelector('#fly-caption').textContent = `Paired illustrative trial. Playback and episode outcomes are labelled in the video; population results appear in Figure ${task.figure}.`;
  updateFly();
}));
function updateFly() { setVideo(document.querySelector('#fly-video'), flyCondition.value, `Paired connectome and MLP fly ${flyTask} experiment: ${flyCondition.selectedOptions[0].textContent}`); }
flyCondition.addEventListener('change', updateFly);
flyChoices.addEventListener('change', event => {
  if (!event.target.matches('input[type="radio"]')) return;
  flyCondition.value = event.target.value;
  updateFly();
});

let maze = 'medium';
const mazeEvidence = {
  umaze: { title: 'U-Maze success', adult: 84.8, mlp: 81.5, text: 'Success rates remain similar in this simpler layout.', sample: '20 paired training seeds · 100 matched evaluation resets per seed' },
  medium: { title: 'Medium-maze success', adult: 64.2, mlp: 44.1, text: 'Success reaches <strong>50.6% vs 20.4%</strong> on routes requiring six or seven turns.', sample: '20 paired training seeds · 100 matched evaluation resets per seed' },
  large: { title: 'Large-maze navigation', text: 'The connectome controller achieves the <strong>highest mean success</strong> among the seven evaluated architectures.', sample: 'Adult connectome · Transformer · MLP · LSTM · GRU · Reservoir · Reservoir + readout' }
};
function updateMaze() {
  const all = document.querySelector('#maze-comparison input:checked').value === 'all';
  const stem = all ? `maze_${maze}_all_seven` : `maze-${maze}`;
  setVideo(document.querySelector('#maze-video'), stem, `${all ? 'Seven controllers' : 'Connectome and MLP'} navigate the ${maze === 'umaze' ? 'U-Maze' : maze + ' maze'}`);
  const evidence = mazeEvidence[maze];
  const panel = document.querySelector('.maze-evidence');
  panel.querySelector('h3').textContent = evidence.title;
  const bars = panel.querySelector('.bar-pair');
  bars.hidden = maze === 'large';
  if (maze !== 'large') {
    bars.setAttribute('aria-label', `Adult connectome success ${evidence.adult} percent; MLP success ${evidence.mlp} percent`);
    bars.querySelectorAll('.bar-label strong')[0].textContent = `${evidence.adult}%`;
    bars.querySelectorAll('.bar-label strong')[1].textContent = `${evidence.mlp}%`;
    bars.querySelector('.adult-bar').style.width = `${evidence.adult}%`;
    bars.querySelector('.mlp-bar').style.width = `${evidence.mlp}%`;
  }
  panel.querySelector('p:not(.sample)').innerHTML = evidence.text;
  panel.querySelector('.sample').textContent = evidence.sample;
}
document.querySelectorAll('[data-maze]').forEach(button => button.addEventListener('click', () => { selectTab(button); maze = button.dataset.maze; updateMaze(); }));
document.querySelector('#maze-comparison').addEventListener('change', updateMaze);

const bodies = { cassie: 'Cassie', tripod: 'Radial tripod', go1: 'Unitree Go1', hexapod: 'Hexapod' };
let body = 'cassie';
const bodyCondition = document.querySelector('#body-condition');
const bodyConditions = [['flat','Flat ground'],['push010','Mild side push'],['push020','Strong side push'],['friction','Reduced friction'],['delay','80 ms action delay'],['sensor','25% sensor loss'],['extended','60-second trial']];
function updateBody() {
  const condition = bodyCondition.value;
  const suffix = { push010: 'push_010', push020: 'push_020', friction: 'friction_060', delay: 'action_delay_080', sensor: 'sensor_loss_025', extended: 'flat_60s' };
  const stem = condition === 'flat' ? `robot-${body}` : `robot_${body}_${suffix[condition]}`;
  const task = body === 'cassie' ? 'postural balance' : 'locomotion';
  document.querySelector('#body-video-title').textContent = `${bodies[body]} · simulated ${task}`;
  setVideo(document.querySelector('#body-video'), stem, `Connectome and MLP ${bodies[body]} ${task}: ${bodyCondition.selectedOptions[0].textContent}`);
}
document.querySelectorAll('[data-body]').forEach(button => button.addEventListener('click', () => {
  selectTab(button); body = button.dataset.body;
  options(bodyCondition, bodyConditions, bodyCondition.value);
  updateBody();
}));
bodyCondition.addEventListener('change', updateBody);

const figureTitles = ['Control architecture','Terrain adaptation','Turning and ascending pathways','Speed control','Ant locomotion and navigation','Transfer across bodies'];
const figureDescriptions = [
  'Fixed adult Drosophila brain connectivity supports a neuromechanical fly and simulated bodies through learned sensory and motor interfaces. The framework spans terrain negotiation, turning, speed control, Ant navigation and four additional embodiments.',
  'Terrain adaptation across block heights, with 20 paired training seeds and 64 shared phases per seed and height. Neural sensitivity and graded mechanosensory blockade probe information routing. The paired 0.525-mm crossing is an outcome-selected example.',
  'Turning experiments across 24 paired training seeds. Command roughness and variation quantify smoothness; terminal heading error measures accuracy. Neural sensitivity, activity enrichment and ascending-to-descending pathway cuts probe the controller. The pathway intervention measures heading error.',
  'Speed adaptation to ramp and sinusoidal commands across 20 paired training seeds. Behavioral tracking measures are paired with neural sensitivity and population interventions. Local policy KL quantifies output sensitivity around a learned policy.',
  'Ant flat-ground locomotion and navigation across U-Maze, Medium and Large layouts. Seven architectures are compared in panel e. Detailed Adult–MLP navigation analyses use 20 paired training seeds and 100 matched evaluation resets per seed. Flat-ground speed gains accompany a higher episode-failure fraction.',
  'Cassie, radial tripod, Unitree Go1 and hexapod evaluations use 20 paired training seeds and 20 matched evaluation states per condition. Normalized performance is standing-duration fraction for Cassie and walking success for the other bodies. Snapshots depict selected paired trials.'
];
const dialog = document.querySelector('#figure-dialog');
let previousFocus;
document.addEventListener('click', event => {
  const button = event.target.closest('[data-figure]');
  if (!button) return;
  const number = Number(button.dataset.figure);
  const methodPanel = button.dataset.panel === 'method';
  previousFocus = button;
  document.querySelector('#figure-dialog-title').textContent = methodPanel ? 'Figure 6b · Shared core, learned interfaces' : `Figure ${number} · ${figureTitles[number - 1]}`;
  const src = methodPanel ? 'assets/images/method-interfaces.webp' : `assets/figures/figure-${number}.webp`;
  document.querySelector('#figure-image').src = src;
  document.querySelector('#figure-image').alt = methodPanel ? 'Shared fixed connectome with learned sensory and motor interfaces, local feedback and four simulated body plans' : `Figure ${number}: ${figureTitles[number - 1]}`;
  document.querySelector('#figure-original').href = src;
  document.querySelector('#figure-description').textContent = methodPanel ? 'Original architecture panel from Figure 6. The recurrent fly-connectome core stays fixed while body-specific sensory and motor interfaces are learned. State feedback and motor actions close the loop through each simulated body.' : figureDescriptions[number - 1];
  document.querySelector('.figure-scroll').classList.remove('zoomed');
  dialog.showModal();
  document.body.style.overflow = 'hidden';
});
document.querySelector('#figure-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
dialog.addEventListener('close', () => { document.body.style.overflow = ''; previousFocus?.focus({ preventScroll: true }); });
document.querySelector('#figure-image').addEventListener('click', () => document.querySelector('.figure-scroll').classList.toggle('zoomed'));

if (!reducedMotion.matches) {
  const revealObserver = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) { entry.target.classList.remove('pending'); revealObserver.unobserve(entry.target); }
  }, { threshold: 0.08 });
  document.querySelectorAll('.section-intro, .method-layout, .circuit-grid, .body-results, .figure-grid').forEach(element => {
    element.classList.add('reveal', 'pending'); revealObserver.observe(element);
  });
}
