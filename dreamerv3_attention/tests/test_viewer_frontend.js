const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync('dreamerv3_attention/viewer.html', 'utf8');

function functionSource(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `${name} should exist in viewer.html`);
  const open = html.indexOf('{', start);
  let depth = 0;
  for (let index = open; index < html.length; index += 1) {
    if (html[index] === '{') depth += 1;
    if (html[index] === '}') depth -= 1;
    if (depth === 0) return html.slice(start, index + 1);
  }
  throw new Error(`Could not find the end of ${name}`);
}

function element() {
  const item = {
    children: [], textContent: '', hidden: false, colSpan: 0,
    classTokens: [],
    appendChild(child) { this.children.push(child); this.lastChild = child; },
    replaceChildren(...children) { this.children = children; },
  };
  item.classList = { add(token) { item.classTokens.push(token); } };
  return item;
}

const body = element();
const footer = element();
const total = element();
const ids = { rewardRows: body, rewardTotal: footer, rewardTotalValue: total };
const document = { createElement: () => element() };
const context = {
  document,
  $: id => ids[id],
  number: (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : '—',
  learnerModeCache: { run: null, mode: null },
};
vm.createContext(context);
vm.runInContext(`${functionSource('renderRewards')}\nthis.renderRewards=renderRewards;`, context);
vm.runInContext(`${functionSource('learnerModeText')}\nthis.learnerModeText=learnerModeText;`, context);

context.renderRewards({
  reward_breakdown: [
    { name: '<b>Waypoint</b>', value: 'true', reward: 50 },
    { name: 'Red light', value: '2 active steps', reward: -100 },
  ],
  episode_return: -50,
});
assert.equal(body.children.length, 2);
assert.equal(body.children[0].children[0].textContent, '<b>Waypoint</b>');
assert.equal(body.children[0].lastChild.classTokens[0], 'positive');
assert.equal(body.children[1].lastChild.classTokens[0], 'negative');
assert.equal(total.textContent, '-50.000');
assert.equal(footer.hidden, false);

context.renderRewards({ reward_breakdown: [], episode_return: null });
assert.equal(body.children.length, 1);
assert.equal(body.children[0].children[0].textContent, 'No reward data for this episode yet.');
assert.equal(footer.hidden, true);

const state = { run: 'run-a', training_mode: 'dreamer' };
assert.equal(context.learnerModeText(state, { updates: 1, learner_mode: 'world_model' }), 'world model');
assert.equal(context.learnerModeText(state, { updates: 1 }), 'world model');
assert.equal(context.learnerModeText({ ...state, run: 'run-b' }, { updates: 0 }), 'warming up');
assert.equal(context.learnerModeText({ ...state, run: 'run-c', training_mode: 'collection' }, {}),
  'collection only');
