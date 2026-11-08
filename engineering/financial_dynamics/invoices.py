"""Draft-only AUD invoice calculator; no issuance, payment or tax filing.

Checksum validation is NOT registry verification. Supply registration evidence
and entity/account mapping in the private accounting service before issuance.
"""
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import date
import hashlib
import html
import json


def _abn_valid(abn):
    if not isinstance(abn,str) or len(abn)!=11 or not abn.isascii() or not abn.isdigit():
        return False
    digits=list(map(int,abn))
    digits[0]-=1
    return sum(a*b for a,b in zip(digits,[10,1,3,5,7,9,11,13,15,17,19]))%89==0


def prepare_invoice(number,seller,abn,gst_registered,lines,buyer=None,issue_date=None):
    if type(gst_registered) is not bool:
        raise ValueError('GST registration status must be known')
    if not number or not seller or not lines:
        raise ValueError('invoice identity and lines required')
    if gst_registered and not _abn_valid(abn):
        raise ValueError('valid ABN required for tax invoice draft')
    if issue_date is not None:
        date.fromisoformat(issue_date)
    cent=Decimal('.01')
    rows=[]
    for line in lines:
        try:
            q=Decimal(str(line['quantity']))
            price=Decimal(str(line['unit_price']))
        except (KeyError,InvalidOperation) as e:
            raise ValueError('invalid line values') from e
        if not q.is_finite() or not price.is_finite() or q<=0 or price<0 or not line.get('description') or type(line.get('taxable')) is not bool:
            raise ValueError('invalid invoice line')
        if line['taxable'] and not gst_registered:
            raise ValueError('GST cannot be charged without registration')
        net=(q*price).quantize(cent,rounding=ROUND_HALF_UP)
        gst=(net*Decimal('.10')).quantize(cent,rounding=ROUND_HALF_UP) if line['taxable'] else Decimal('0.00')
        rows.append({'description':str(line['description']),'quantity':str(q),'unit_price':str(price),
                     'taxable':line['taxable'],'net':str(net),'gst':str(gst),'total':str(net+gst)})
    net=sum(Decimal(row['net']) for row in rows)
    gst=sum(Decimal(row['gst']) for row in rows)
    if gst_registered and net+gst>=1000 and not buyer:
        raise ValueError('buyer identity or ABN required for tax invoices >= AUD 1000')
    out={'status':'draft','number':str(number),'seller':str(seller),'abn':abn,'buyer':buyer,
         'title':'Tax invoice' if gst_registered else 'Invoice','currency':'AUD',
         'issue_date':issue_date,'lines':rows,'net':str(net),'gst':str(gst),'total':str(net+gst),
         'registration_verified':False}
    out['content_sha256']=hashlib.sha256(json.dumps(out,sort_keys=True).encode()).hexdigest()
    return out


def render_draft(invoice):
    esc=lambda x:html.escape(str(x if x is not None else 'Not configured'))
    rows=''.join('<tr>'+''.join('<td>'+esc(row[k])+'</td>' for k in ['description','quantity','unit_price','taxable','net','gst','total'])+'</tr>' for row in invoice['lines'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Draft invoice</title>'
            '<style>body{font:16px sans-serif;max-width:800px;margin:40px auto}td,th{padding:10px;text-align:left}table{border-collapse:collapse;width:100%}td{border-top:1px solid #ccc}</style>'
            '<h1>DRAFT — '+esc(invoice['title'])+'</h1><p>'+esc(invoice['seller'])+'<br>ABN: '+esc(invoice['abn'])+'</p>'
            '<p>Number: '+esc(invoice['number'])+'<br>Date: '+esc(invoice['issue_date'])+'<br>Buyer: '+esc(invoice['buyer'])+'</p>'
            '<table><tr><th>Description</th><th>Quantity</th><th>Unit price</th><th>Taxable</th><th>Net</th><th>GST</th><th>Total</th></tr>'+rows+'</table>'
            '<p>Total AUD '+esc(invoice['total'])+' · GST '+esc(invoice['gst'])+'</p>'
            '<p>Draft only. Entity and registration evidence must be checked before issuance.</p></html>')
