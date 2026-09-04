## Purpose

Detection of PII in free-form Russian text that both the LLM text pipeline
and the PDF anonymizer rely on to find what to mask, going beyond the
regex/pattern layer to structured entities like postal addresses that
regex alone cannot reliably delimit.

## ADDED Requirements

### Requirement: Detection of structured postal address components
The system SHALL detect each structured Russian postal address component
present in input text — postal index, city, region, district, street,
house, building, office, and apartment/room — as its own PII span, for
every such component found, not only the city or street name.

#### Scenario: Address with index, city, street, and house number
- **WHEN** input text contains a postal address with an index, city,
  street, and house number (e.g. "620004, г. Екатеринбург, ул. Малышева,
  101")
- **THEN** the detection result includes a separate PII span for the
  index, for the city, for the street, and for the house number

#### Scenario: Address with building and office qualifiers
- **WHEN** input text contains a postal address that includes building
  and office/room qualifiers (e.g. "д 68/70 стр 1, офис 54")
- **THEN** the detection result includes a PII span covering the building
  qualifier and a separate PII span covering the office/room qualifier,
  in addition to the spans for the preceding address components

### Requirement: House number detection without a full marker word
A house number immediately following a detected street component SHALL be
detected as its own PII span even when it has no marker word at all, or
only an abbreviated marker without a period, rather than the full `д.`
(with period) or `дом` form.

#### Scenario: House number with no marker at all
- **WHEN** input text contains a street followed directly by a bare
  number with no marker word (e.g. "ул. Малышева, 101")
- **THEN** the detection result includes a PII span covering that number
  in addition to the span for the street

#### Scenario: House number with an unpunctuated abbreviated marker
- **WHEN** input text contains a street followed by a house number using
  the abbreviated marker without a period (e.g. "ул Бутырский Вал,
  д 68/70")
- **THEN** the detection result includes a PII span covering the house
  number in addition to the span for the street

### Requirement: Room/premises qualifier detected after a house number
A room or premises qualifier ("помещ." or "помещение", with a number)
immediately following a detected house number SHALL be detected as its
own PII span, even though no such qualifier type is otherwise recognized
by the underlying address grammar.

#### Scenario: Room qualifier after a house number
- **WHEN** input text contains a house number followed directly by a
  room/premises qualifier and number (e.g. "д. 14, помещ. 1/1")
- **THEN** the detection result includes a PII span covering the room
  qualifier in addition to the span for the house number

### Requirement: Independent handling of multiple addresses in one text
When input text contains more than one distinct postal address, the
system SHALL detect each address's components independently, without
merging the components of separate addresses — or any non-address text
that separates them — into a single span.

#### Scenario: Two organizations' addresses in the same paragraph
- **WHEN** input text contains two distinct organizations, each with its
  own postal address, separated by other text (e.g. a second
  organization's name and its own preceding label)
- **THEN** the detection result's address-component spans for the first
  address end before the intervening non-address text, and the spans for
  the second address start after it — none of the intervening text is
  included in any address-component span

### Requirement: Graceful degradation on address-extraction failure
A failure while extracting address components from a given input SHALL
NOT raise out of detection or discard results already produced by other
detection layers for that input.

#### Scenario: Address extraction raises on malformed input
- **WHEN** address-component extraction raises an exception while
  processing a given text
- **THEN** detection for that text still completes and returns whatever
  spans the pattern-based and general NER layers found, without
  propagating the exception
