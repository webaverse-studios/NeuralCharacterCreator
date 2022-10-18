# NeuralCharacterCreator
Create a character based on a text description, using assets from the Webaverse Character Creator.

Uses GPT-3 for entity extraction, followed by cosine similiarity matching using tensorflow.js

Data is pulled from dataset.json -- embedding takes ~30 seconds so we are caching.

To test:
```
npm run start -- --name Hyacinth --description "Beastmage. She wears a shirt with a cute cat on it, a fine white labcoat, combat boots, and she carries a giant mace which she bludgeons her enemies with."
```

Extracted Traits:
```
{
  body: 'Shirt with a cute cat on it',
  chest: 'Dirty labcoat',
  feet: 'Combat boots',
  hands: 'none',
  head: 'none',
  legs: 'none',
  neck: 'none',
  waist: 'none',
  weapon: 'Giant mace'
}
```

Output items from search
```
{
  body: 'Shirt',
  chest: 'Ornate Chestplate',
  feet: 'Demonhide Boots',
  hands: null,
  head: null,
  legs: null,
  neck: null,
  waist: null,
  weapon: 'Short Sword'
}
```
