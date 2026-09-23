# Arabic text spot-check: 20 articles

Each article's `text_ar` below needs to be checked against the actual source PDF page it cites. Open `data/raw/civil_code.pdf` to the noted page, compare, and tick PASS or FAIL with a note.

| # | Article | Page | Reason for inclusion | text_ar preview | Result |
|---|---------|------|----------------------|------------------|--------|
| 1 | 1 | 1 | first article in the corpus (also the promulgation-law duplicate-numbering edge case) | (١) تسرى النصوص التشريعية على جميع المسائل التي تتناولها هذه... | [ ] PASS / [ ] FAIL |
| 2 | 60 | 7 | inside the 54-80 repealed range (deduped placeholder text) | ألغيت المواد من ٥٤ إلى ٨٠ بالقرار الجمهوري بالقانون رقم ٣٤٨ ... | [ ] PASS / [ ] FAIL |
| 3 | 84 | 8 | stratified random sample (seed=42) across the full article range | (١( األشياء القابلة لالستهالك هي التي ينحصر استعمالها، بحسب ... | [ ] PASS / [ ] FAIL |
| 4 | 112 | 11 | stratified random sample (seed=42) across the full article range | إذا بلغ الصبى المميز الثامنة عشرة من عمره واذن له فى تسلم أم... | [ ] PASS / [ ] FAIL |
| 5 | 147 | 16 | the running example used throughout this project's own schema design | ( ١ ( العقد شريعة المتعاقدين ، فال يجوز نقضه وال تعديله إال ... | [ ] PASS / [ ] FAIL |
| 6 | 197 | 24 | stratified random sample (seed=42) across the full article range | تسقط الدعوى الناشئة عن الفضالة بانقضاء ثالث سنوات من اليوم ا... | [ ] PASS / [ ] FAIL |
| 7 | 238 | 30 | paragraph-split article (Arabic split by numbered clause, English kept whole) | (١( إذا كان تصرف المدين بعوض، اشترط لعدم نفاذه في حق الدائن ... | [ ] PASS / [ ] FAIL |
| 8 | 384 | 52 | stratified random sample (seed=42) across the full article range | (١( ينقطع التقادم إذا أقر المدين بحق الدائن إقرارا صريحا أو ... | [ ] PASS / [ ] FAIL |
| 9 | 400 | 53 | inside the 389-417 repealed range (headerless repeal block, no article marker in source) | المواد من ٣٨٩ إلى ٤١٧ ملغاة... | [ ] PASS / [ ] FAIL |
| 10 | 421 | 53 | stratified random sample (seed=42) across the full article range | (١( في البيع بشرط التجربة يجوز للمشتري أن يقبل المبيع أو برف... | [ ] PASS / [ ] FAIL |
| 11 | 512 | 67 | stratified random sample (seed=42) across the full article range | (١( إذا تعهد الشريك بأن يقدم حصته في الشركة عمال وجب عليه أن... | [ ] PASS / [ ] FAIL |
| 12 | 604 | 82 | stratified random sample (seed=42) across the full article range | )١( إذا انتقلت ملكية العين المؤجرة اختيارا أو جب ًرا إلى شخص... | [ ] PASS / [ ] FAIL |
| 13 | 658 | 89 | paragraph-split article (4 clauses, Arabic split, English kept whole) | (١( إذا أبرم العقد بأجر إجمالي على أساس تصميم أتفق عليه رب ا... | [ ] PASS / [ ] FAIL |
| 14 | 689 | 96 | stratified random sample (seed=42) across the full article range | يجب على العامل إلى جانب االلتزامات المبينة فى المواد السابقة... | [ ] PASS / [ ] FAIL |
| 15 | 861 | 123 | stratified random sample (seed=42) across the full article range | ال يجوز لصاحب العلو أن يزيد فى ارتفاع بنائه بحيث يضر بالسفل.... | [ ] PASS / [ ] FAIL |
| 16 | 875 | 125 | stratified random sample (seed=42) across the full article range | (١( تعيين الورثة وتحديد أنصبائهم فى اإلرث وانتقال أموالهم ال... | [ ] PASS / [ ] FAIL |
| 17 | 1022 | 147 | source PDF has a genuinely empty Arabic cell for this article -- manually patched | (١) نفقة الأعمال اللازمة لاستعمال حق الارتفاق وللقيام بحفظه ... | [ ] PASS / [ ] FAIL |
| 18 | 1044 | 151 | stratified random sample (seed=42) across the full article range | للراهن الحق فى إدارة العقار المرهون وفى قبض ثماره إلى وقت ال... | [ ] PASS / [ ] FAIL |
| 19 | 1147 | 170 | stratified random sample (seed=42) across the full article range | (١) ما يستحق لبائع العقار من الثمن وملحقاته ، يكون له امتياز... | [ ] PASS / [ ] FAIL |
| 20 | 1149 | 170 | last article in the corpus | للشركاء الذين اقتسموا عقاراً ، حق امتياز عليه تأمينا لما تخو... | [ ] PASS / [ ] FAIL |

## Notes

(add any discrepancies found here, with article number and what was wrong)
